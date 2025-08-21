# Requires: tensorflow >= 2.x
import math
import os
import queue
import shutil
import signal
import time

import numpy as np
from tensorflow.data import AUTOTUNE

from rw_lock import RWLock
from src.model import NavigationModel

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from tensorflow.keras import layers

from simulation import read_player_states, send_model_outputs

import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"

from src.hyperparameters import *

CURRENT_MODEL_PATH = "models/model-current.keras"
PREVIOUS_MODEL_PATH = "models/model-previous.keras"

# instantiate train and prediction model
nav_model = NavigationModel(CURRENT_MODEL_PATH)
model_lock = RWLock()
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

# ------------------------
# Utils: probability/log-prob for multi-binary actions (Bernoulli)
# ------------------------
def bernoulli_log_probs(probs, actions):
    """
    probs: (batch, ... , NUM_ACTIONS) values in (0,1)
    actions: same shape, 0/1
    returns: log_prob summed over action dims -> shape (batch, ...)
    """
    eps = 1e-8
    logp = actions * tf.math.log(probs + eps) + (1.0 - actions) * tf.math.log(1.0 - probs + eps)
    return tf.reduce_sum(logp, axis=-1)


# ------------------------
# GAE advantage computation
# ------------------------
def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=LAMBDA):
    """
    rewards: (T, Nenv)
    values: (T, Nenv, SEQ_LEN)
    dones: (T, Nenv) 1 if episode ended after step t, else 0
    last_value: (Nenv, SEQ_LEN) bootstrap value for step T
    returns: advantages (T, Nenv), returns (T, Nenv)
    """
    T = rewards.shape[0]
    N = rewards.shape[1]
    advantages = np.zeros((T, N), dtype=np.float32)
    last_adv = np.zeros(N, dtype=np.float32)
    # iterate reversed
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = last_value
            next_nonterminal = 1.0 - dones[t]
        else:
            next_val = values[t + 1]
            next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        last_adv = delta + gamma * lam * next_nonterminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


# PPO update for collected rollout
# SHOULD ONLY BE CALLED IN THE MODEL TRAINING THREAD LOOP.
# updating training model is not thread safe
# The train_step is compiled and expects tensors. It performs forward/backward
# and apply_gradients inside the TF graph to reduce Python overhead.
@tf.function
def _train_step(mb_ang, mb_lin, mb_rot, mb_laser_dist, mb_laser_type,
                mb_actions, mb_old_logp, mb_adv, mb_returns, mb_h0, mb_c0):
    """
    returns: model monitoring
    """
    # mb_pos: (mb, T, 3), mb_actions: (mb, T, NUM_ACTIONS), mb_old_logp: (mb, T)
    with tf.GradientTape() as tape:
        policy_seq, values_seq_pred, _, _ = nav_model.forward_sequence(
            mb_ang, mb_lin, mb_rot, mb_laser_dist, mb_laser_type, h0=mb_h0, c0=mb_c0
        )
        # policy_seq: (mb, T, NUM_ACTIONS), values_seq_pred: (mb, T)

        probs_flat = tf.reshape(policy_seq, (-1, NUM_ACTIONS))  # (mb*T, NUM_ACTIONS)
        actions_flat = tf.reshape(mb_actions, (-1, NUM_ACTIONS))
        logp_flat = bernoulli_log_probs(probs_flat, actions_flat)  # (mb*T,)
        logp = tf.reshape(logp_flat, (tf.shape(mb_actions)[0], tf.shape(mb_actions)[1]))  # (mb, T)

        ratio = tf.exp(logp - mb_old_logp)
        unclipped = ratio * mb_adv
        clipped = tf.clip_by_value(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
        policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

        value_loss = VALUE_COEFF * tf.reduce_mean(tf.square(mb_returns - values_seq_pred))

        eps = 1e-8
        entropy_per_action = -(policy_seq * tf.math.log(policy_seq + eps) +
                               (1 - policy_seq) * tf.math.log(1 - policy_seq + eps))
        entropy = tf.reduce_mean(entropy_per_action)
        entropy_loss = -ENTROPY_COEFF * entropy

        total_loss = policy_loss + value_loss + entropy_loss

    grads = tape.gradient(total_loss, nav_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, nav_model.trainable_variables))




def ppo_update_tf(rollout):
    """
    Optimized PPO update that:
      - Converts entire rollout to TF tensors once
      - Builds a tf.data.Dataset of environments (N examples, each is a full T-sequence)
      - Uses a compiled `_train_step` for minibatch updates

    This function keeps behaviour compatible with your original shapes.
    """
    # Convert rollout format from (T, N, ...) to (N, T, ...)
    ang_seq = np.transpose(rollout['angvel'], (1, 0, 2))
    lin_seq = np.transpose(rollout['linvel'], (1, 0, 2))
    rot_seq = np.transpose(rollout['rot'], (1, 0, 2))
    laser_dist_seq = np.transpose(rollout['laser_dist'], (1, 0, 2, 3))  # (N,T,LASERS,1)
    laser_type_seq = np.transpose(rollout['laser_type'], (1, 0, 2)).astype(np.int32)
    actions_seq = np.transpose(rollout['actions'], (1, 0, 2))  # (N,T,NUM_ACTIONS)
    old_logp_seq = np.transpose(rollout['old_logp'], (1, 0))  # (N,T)

    # compute last_value for bootstrap using train_model
    last_ang = rollout['angvel'][-1]
    last_lin = rollout['linvel'][-1]
    last_rot = rollout['rot'][-1]
    last_laser_dist = rollout['laser_dist'][-1]
    last_laser_type = rollout['laser_type'][-1].astype(np.int32)

    # model_lock.acquire_read()
    # Convert final states to tensors and call model once
    last_policy_tf, last_value_tf, _, _ = nav_model.step(
        tf.convert_to_tensor(last_ang), tf.convert_to_tensor(last_lin),
        tf.convert_to_tensor(last_rot), tf.convert_to_tensor(last_laser_dist), tf.convert_to_tensor(last_laser_type),
        tf.convert_to_tensor(rollout['h_last']), tf.convert_to_tensor(rollout['c_last'])
    )
    # model_lock.release_read()
    last_value = last_value_tf.numpy()  # (N,)

    advantages, returns = compute_gae(rollout['rewards'], rollout['values'], rollout['dones'], last_value, gamma=GAMMA,
                                      lam=LAMBDA)

    adv_seq = np.transpose(advantages, (1, 0))  # (N,T)
    returns_seq = np.transpose(returns, (1, 0))  # (N,T)

    # Convert everything once to TF tensors (shapes: N, T, ...)
    ang_tf = tf.convert_to_tensor(ang_seq)
    lin_tf = tf.convert_to_tensor(lin_seq)
    rot_tf = tf.convert_to_tensor(rot_seq)
    laser_dist_tf = tf.convert_to_tensor(laser_dist_seq)
    laser_type_tf = tf.convert_to_tensor(laser_type_seq)
    actions_tf = tf.convert_to_tensor(actions_seq)
    old_logp_tf = tf.convert_to_tensor(old_logp_seq)
    adv_tf = tf.convert_to_tensor(adv_seq)
    returns_tf = tf.convert_to_tensor(returns_seq)
    h0_tf = tf.convert_to_tensor(rollout['h0'])
    c0_tf = tf.convert_to_tensor(rollout['c0'])

    # normalize advantage
    adv_tf = (adv_tf - tf.reduce_mean(adv_tf)) / (tf.math.reduce_std(adv_tf) + 1e-8)

    # Build dataset: each example is one environment's full T-sequence
    ds = tf.data.Dataset.from_tensor_slices(
        (ang_tf, lin_tf, rot_tf, laser_dist_tf, laser_type_tf,
         actions_tf, old_logp_tf, adv_tf, returns_tf, h0_tf, c0_tf)
    )

    num_samples = ang_seq.shape[0]

    # Training loop: shuffle per-epoch and batch
    for epoch in range(PPO_EPOCHS):
        # print(f"Epoch {epoch + 1}")
        # shuffle and batch per epoch to get different minibatches each epoch
        ds_epoch = ds.shuffle(buffer_size=max(1024, num_samples), reshuffle_each_iteration=True)
        ds_epoch = ds_epoch.batch(MINIBATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

        for batch in ds_epoch:
            (mb_ang, mb_lin, mb_rot, mb_laser_dist, mb_laser_type,
             mb_actions, mb_old_logp, mb_adv, mb_returns, mb_h0, mb_c0) = batch

            # model_lock.acquire_write()
            # call compiled train step
            _train_step(mb_ang, mb_lin, mb_rot, mb_laser_dist, mb_laser_type,
                                           mb_actions, mb_old_logp, mb_adv, mb_returns, mb_h0, mb_c0)

            # model_lock.release_write()

    return


# ------------------------
# Rollout collector (interacts with simulation)
# ------------------------
# -- Placeholder functions for sim interaction: replace with your IPC/FFI/pyO3 calls.
def sim_get_states(conn):
    """
    Replace with code that returns per-player states from the simulation.
    Should return dictionaries of numpy arrays with shape (Nplayers, ...).
    Example returns:
      {
        "position": np.array(shape=(N,3)),
        "ang_velocity": np.array((N,3)),
        "lin_velocity": np.array((N,3)),
        "rotation": np.array((N,3)),
        "lasers": { "dist": np.array((N, LASERS, 1)), "type": np.array((N, LASERS)) }
      }
    """
    return read_player_states(conn)


def sim_send_actions(conn, actions_per_player):
    """
    Send actions back to sim. actions_per_player: numpy array shape (N, NUM_ACTIONS) 0/1
    """
    # Ensure it's integers
    arr = actions_per_player.astype(int)

    # Compute powers of 2 for each bit (assuming most significant bit first)
    powers_of_two = 2 ** np.arange(arr.shape[1] - 1, -1, -1)

    # Multiply and sum along axis 1
    actions_per_player = arr.dot(powers_of_two)
    return send_model_outputs(conn, actions_per_player)



def collect_rollout(conn, rollout_length=SEQ_LEN):
    """
    Collect a rollout of length T for all N players currently in sim.
    Returns the 'rollout' dict used by ppo_update.
    """
    # initial read to get number of players N and initial states
    sim_state = sim_get_states(conn)
    N = sim_state['position'].shape[0]
    T = rollout_length

    # buffers (T, N, ...)
    ang_buf = np.zeros((T, N, 3), dtype=np.float32)
    lin_buf = np.zeros((T, N, 3), dtype=np.float32)
    rot_buf = np.zeros((T, N, 3), dtype=np.float32)
    laser_dist_buf = np.zeros((T, N, LASERS_PER_PLAYER, 1), dtype=np.float32)
    laser_type_buf = np.zeros((T, N, LASERS_PER_PLAYER), dtype=np.int32)

    actions_buf = np.zeros((T, N, NUM_ACTIONS), dtype=np.float32)
    old_logp_buf = np.zeros((T, N), dtype=np.float32)
    values_buf = np.zeros((T, N), dtype=np.float32)
    rewards_buf = np.zeros((T, N), dtype=np.float32)
    dones_buf = np.zeros((T, N), dtype=np.float32)

    # initial LSTM states per env
    h = np.zeros((N, LSTM_UNITS), dtype=np.float32)
    c = np.zeros((N, LSTM_UNITS), dtype=np.float32)

    # also store initial states to use at training time
    h0 = h.copy()
    c0 = c.copy()

    for t in range(T):
        ang = sim_state['angvel']
        lin = sim_state['linvel']
        rot = sim_state['rotation']
        laser_dist = sim_state['laser']['distance']
        laser_type = sim_state['laser']['type'].astype(np.int32)

        # store state
        ang_buf[t] = ang
        lin_buf[t] = lin
        rot_buf[t] = rot
        laser_dist_buf[t] = laser_dist
        laser_type_buf[t] = laser_type

        # model_lock.acquire_read()
        # run model step (batch)
        policy_tf, value_tf, h_tf, c_tf = nav_model.step(
            tf.convert_to_tensor(ang),
            tf.convert_to_tensor(lin), tf.convert_to_tensor(rot),
            tf.convert_to_tensor(laser_dist), tf.convert_to_tensor(laser_type),
            tf.convert_to_tensor(h), tf.convert_to_tensor(c)
        )
        # model_lock.release_read()
        policy = policy_tf.numpy()  # (N, NUM_ACTIONS)
        value = value_tf.numpy()
        h = h_tf.numpy()
        c = c_tf.numpy()
        # Sample actions from bernoulli (we use probabilities policy)
        actions = (np.random.rand(*policy.shape) < policy)  # (N, SEQ_LEN, NUM_ACTIONS,)
        # compute log_probs under the policy we used (store old_logp for PPO)
        eps = 1e-8
        logp = np.sum(actions * np.log(policy + eps) + (1 - actions) * np.log(1 - policy + eps),
                      axis=-1)  # (N, SEQ_LEN)

        # send actions to simulation
        sim_send_actions(conn, actions)

        sim_state = sim_get_states(conn)  # compute from the next state the rewards, dones etc

        rewards = sim_state["reward"]
        dones = sim_state["done"]

        # store action / logging info
        actions_buf[t] = actions
        old_logp_buf[t] = logp
        values_buf[t] = value
        rewards_buf[t] = rewards
        dones_buf[t] = dones

        # reset LSTM states where done==1
        for i in range(N):
            if dones[i]:
                h[i] = np.zeros(LSTM_UNITS, dtype=np.float32)
                c[i] = np.zeros(LSTM_UNITS, dtype=np.float32)
    rollout = {
        'angvel': ang_buf,
        'linvel': lin_buf,
        'rot': rot_buf,
        'laser_dist': laser_dist_buf,
        'laser_type': laser_type_buf,
        'actions': actions_buf,
        'old_logp': old_logp_buf,
        'values': values_buf,
        'rewards': rewards_buf,
        'dones': dones_buf,
        'h0': h0,
        'c0': c0,
        'h_last': h,  # final states after last step
        'c_last': c
    }
    return rollout


stop = False


def handle_signal(s, f):
    global stop
    stop = True
    save_model("model-interrupted.keras")
    exit(1)


def save_model(filename: str):
    model_file = "models/" + filename
    nav_model.save(model_file)
    print(f"Model saved at {model_file}")


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def analyze_rollout(rollout: dict):
    avg_reward = np.mean(rollout["rewards"])
    N = rollout["h0"].shape[0]
    print("Rollout N =", N)
    print("Reward average: {}".format(avg_reward))
    return avg_reward


def handle_collapse(current_update_step):
    print("[Warning] MODEL DID COLLAPSE")
    print("[Info] Model will be restored to its last version.")
    print(f"[Info] LR will be divided by {LR_MULTIPLIER_PER_COLLAPSE_DETECTION} for {REQUIRED_STABLE_UPDATES_AFTER_COLLAPSE} steps")

    # model_lock.acquire_write()

    # Rollback to last viable version.
    # Restore the last saved model (CURRENT_MODEL_PATH)
    # If we're in a step that just precedes a saving, it means that the last saved model (CURRENT_MODEL_PATH) is also dead.
    # So we take the one before it (PREVIOUS_MODEL_PATH).
    nav_model.load(CURRENT_MODEL_PATH if (current_update_step - 1) % BACKUP_RATE != 0 else PREVIOUS_MODEL_PATH)

    # also reduce the LR
    optimizer.learning_rate.assign(optimizer.learning_rate * LR_MULTIPLIER_PER_COLLAPSE_DETECTION)

    # model_lock.release_write()

def reset_default_hyperparameters():
    optimizer.learning_rate.assign(LR)
    print("[Info] Restoring default hyperparameters.")

def save_and_roll_model():
    """
    Copies current model file to previous model file, and
    Saves current model to current model file
    """
    if os.path.exists(CURRENT_MODEL_PATH):
        shutil.move(CURRENT_MODEL_PATH, PREVIOUS_MODEL_PATH)
    nav_model.save(CURRENT_MODEL_PATH)
    print(f"Saved and rolled model.")


def agent_loop(rollout_queue: queue.Queue):


    current_update_step = 0

    # last step when the model collapsed.
    last_collapse_step = 0

    last_reward = math.nan

    while True:
        current_update_step += 1
        try:
            if stop:
                return

            # merge all rollouts
            merged_rollout = rollout_queue.get()  # at least one rollout
            while rollout_queue.qsize() > 0:
                rollout = rollout_queue.get_nowait()
                if not rollout:
                    continue

                for k, v in rollout.items():
                    if k not in merged_rollout:
                        merged_rollout[k] = v
                    else:
                        merged_rollout[k] = np.concatenate([merged_rollout[k], v],
                                                           axis=0 if k in ["h0", "c0", "h_last", "c_last"] else 1)

            reward_avg = analyze_rollout(merged_rollout)

            # if rewards of this step is EXACTLY the same reward of the last step,
            # it means that there is an extremely high chance that the model collapsed
            if last_reward == reward_avg:
                handle_collapse(current_update_step)
                last_collapse_step = current_update_step
                continue

            last_reward = reward_avg

            if current_update_step - last_collapse_step == REQUIRED_STABLE_UPDATES_AFTER_COLLAPSE:
                reset_default_hyperparameters()

            # train on rollout
            # print("Training on rollouts, this might cause simulations to halt for a moment")
            ppo_update_tf(merged_rollout)

            # logging / saving model checkpoints etc.
            print(f"Finished update {current_update_step}.")
            if current_update_step % BACKUP_RATE == 0:
                save_and_roll_model()

        except Exception as e:
            print(e)
            save_model("model-error.keras")
            # exit(1)
