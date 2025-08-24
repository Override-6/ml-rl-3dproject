# Requires: tensorflow >= 2.x
import logging
import math
import os
import queue
import shutil
import signal

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
nav_model_step_tf = tf.function(nav_model.step,
                                experimental_compile=False)  # cannot compile due to lstm which is not completely supported yet


# ------------------------
# Utils: probability/log-prob for multi-binary actions (Bernoulli)
# ------------------------
def bernoulli_log_probs(probs, actions):
    """
    probs: (batch, ... , NUM_ACTIONS) values in (0,1)
    actions: same shape, 0/1
    returns: log_prob summed over action dims -> shape (batch, ...)
    """
    actions = tf.cast(actions, tf.float32)
    eps = 1e-8
    logp = actions * tf.math.log(probs + eps) + (1.0 - actions) * tf.math.log(1.0 - probs + eps)
    return tf.reduce_sum(logp, axis=-1)


# ------------------------
# GAE advantage computation
# ------------------------
# @tf.function
def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=LAMBDA):
    """
    rewards: (T, Nenv)
    values: (T, Nenv, SEQ_LEN)
    dones: (T, Nenv) 1 if episode ended after step t, else 0
    last_value: (Nenv, SEQ_LEN) bootstrap value for step T
    returns: advantages (T, Nenv), returns (T, Nenv)
    """

    rewards = tf.cast(rewards, tf.float32)
    values = tf.cast(values, tf.float32)
    dones = tf.cast(dones, tf.float32)
    last_value = tf.cast(last_value, tf.float32)

    # Build next_values: next_values[t] = values[t+1] for t < T-1, and last_value for t == T-1
    # values[1:] shape = (T-1, N); tf.expand_dims(last_value, 0) shape = (1, N)
    next_values = tf.concat([values[1:], tf.expand_dims(last_value, axis=0)], axis=0)

    # next_nonterminal = 1 - dones[t]
    next_nonterminal = 1.0 - dones

    # delta_t = r_t + gamma * next_value_t * next_nonterminal_t - value_t
    delta = rewards + gamma * next_values * next_nonterminal - values  # shape (T, N)

    # We'll compute advantages by scanning from last -> first.
    # Reverse along time axis, scan forward, then reverse back.
    delta_rev = tf.reverse(delta, axis=[0])  # (T, N) reversed in time
    nonterm_rev = tf.reverse(next_nonterminal, axis=[0])  # (T, N) reversed

    # initializer: zeros per environment (shape (N,))
    # Use the same dtype as tensors
    init_adv = tf.zeros_like(last_value, dtype=tf.float32)  # shape (N,)

    # scan function: given next_adv (accumulator) and current elems, compute adv_t
    def _scan_fn(acc, elems):
        # elems is a tuple (delta_t_rev, nonterm_t_rev)
        delta_t, nonterm_t = elems
        # acc is advantage_{t+1} (since we're scanning reversed)
        return delta_t + gamma * lam * nonterm_t * acc

    # tf.scan expects elems to be a Tensor or nested structure of tensors with leading dim T
    # We pass a tuple of (delta_rev, nonterm_rev). The returned `adv_rev` will have shape (T, N).
    advs_rev = tf.scan(_scan_fn, (delta_rev, nonterm_rev), initializer=init_adv)

    # advs_rev is in reversed time order -> flip back
    advantages = tf.reverse(advs_rev, axis=[0])  # (T, N)

    returns = advantages + values  # (T, N)

    return advantages, returns


# PPO update for collected rollout
# SHOULD ONLY BE CALLED IN THE MODEL TRAINING THREAD LOOP.
# updating training model is not thread safe
# The train_step is compiled and expects tensors. It performs forward/backward
# and apply_gradients inside the TF graph to reduce Python overhead.
@tf.function
def _train_step(mb_ang, mb_lin, mb_rot, mb_laser_dist, mb_laser_type,
                mb_actions, mb_look_dir, mb_old_logp, mb_adv, mb_returns, mb_h0, mb_c0):
    """
    returns: model monitoring
    """
    # mb_pos: (mb, T, 3), mb_actions: (mb, T, NUM_ACTIONS), mb_old_logp: (mb, T)
    with tf.GradientTape() as tape:
        policy_seq, values_seq_pred, look_seq_pred, _, _ = nav_model.forward_sequence(
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

        huber = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)
        value_error = huber(mb_returns, values_seq_pred)  # shape (mb, T)
        value_loss = 0.25 * tf.reduce_mean(value_error)  # reduce coefficient from 0.5 -> 0.25

        eps = 1e-8
        entropy_per_action = -(policy_seq * tf.math.log(policy_seq + eps) +
                               (1 - policy_seq) * tf.math.log(1 - policy_seq + eps))
        entropy = tf.reduce_mean(entropy_per_action)
        entropy_loss = -ENTROPY_COEFF * entropy

        total_loss = policy_loss + value_loss + entropy_loss

        pred_flat = tf.reshape(look_seq_pred, (-1, 1))
        tgt_flat = tf.reshape(tf.cast(mb_look_dir, tf.float32), (-1, 1))

        # Ensure both are normalized
        pred_flat = tf.math.l2_normalize(pred_flat, axis=-1)
        tgt_flat = tf.math.l2_normalize(tgt_flat, axis=-1)

        # cosine loss = mean(1 - dot(pred, tgt))
        cos_sim = tf.reduce_sum(pred_flat * tgt_flat, axis=-1)
        look_loss = tf.reduce_mean(1.0 - cos_sim)

        # Add to total loss
        total_loss = total_loss + look_loss

    grads = tape.gradient(total_loss, nav_model.trainable_variables)

    # Debug: print loss terms and gradient norms
    # compute global norm but handle None grads
    safe_grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, nav_model.trainable_variables)]
    # grad_norm = tf.linalg.global_norm(safe_grads)
    # tf.print("policy_loss=", policy_loss, "value_loss=", value_loss, "entropy=", entropy, "total_loss=", total_loss,
    #          "grad_norm=", grad_norm)
    # tf.print("mean_return=", tf.reduce_mean(mb_returns),
    #              "mean_value_pred=", tf.reduce_mean(values_seq_pred),
    #              "adv mean/std:", tf.reduce_mean(mb_adv), tf.math.reduce_std(mb_adv))

    optimizer.apply_gradients(zip(safe_grads, nav_model.trainable_variables))


def ppo_update_tf(rollout):
    """
    Optimized PPO update that:
      - Converts entire rollout to TF tensors once
      - Builds a tf.data.Dataset of environments (N examples, each is a full T-sequence)
      - Uses a compiled `_train_step` for minibatch updates

    This function keeps behaviour compatible with your original shapes.
    """
    # Convert rollout format from (T, N, ...) to (N, T, ...)
    ang_seq = tf.transpose(rollout['angvel'], [1, 0, 2])
    lin_seq = tf.transpose(rollout['linvel'], [1, 0, 2])
    rot_seq = tf.transpose(rollout['rot'], [1, 0, 2])
    laser_dist_seq = tf.transpose(rollout['laser_dist'], [1, 0, 2, 3])
    laser_type_seq = tf.transpose(rollout['laser_type'], [1, 0, 2])
    actions_seq = tf.transpose(rollout['actions'], [1, 0, 2])
    look_dir_seq = tf.transpose(rollout['look_dir'], [1, 0, 2])
    old_logp_seq = tf.transpose(rollout['old_logp'], [1, 0])

    # compute last_value for bootstrap using train_model
    last_ang = rollout['angvel'][-1]
    last_lin = rollout['linvel'][-1]
    last_rot = rollout['rot'][-1]
    last_laser_dist = rollout['laser_dist'][-1]
    last_laser_type = rollout['laser_type'][-1]

    # model_lock.acquire_read()
    # Convert final states to tensors and call model once
    last_policy_tf, last_value_tf, last_look_tf, _, _ = nav_model_step_tf(
        last_ang, last_lin,
        last_rot, last_laser_dist, last_laser_type,
        rollout['h_last'], rollout['c_last']
    )
    # model_lock.release_read()
    last_value = last_value_tf

    advantages, returns = compute_gae(rollout['rewards'], rollout['values'], rollout['dones'], last_value, gamma=GAMMA,
                                      lam=LAMBDA)

    adv_seq = np.transpose(advantages, (1, 0))  # (N,T)
    returns_seq = np.transpose(returns, (1, 0))  # (N,T)

    # Convert everything once to TF tensors (shapes: N, T, ...)
    ang_tf = tf.cast(ang_seq, tf.float32)
    lin_tf = tf.cast(lin_seq, tf.float32)
    rot_tf = tf.cast(rot_seq, tf.float32)
    laser_dist_tf = tf.cast(laser_dist_seq, tf.float32)
    laser_type_tf = tf.cast(laser_type_seq, tf.int32)
    actions_tf = tf.cast(actions_seq, tf.int32)
    look_dir_tf = tf.cast(look_dir_seq, tf.float32)
    old_logp_tf = tf.cast(old_logp_seq, tf.float32)
    adv_tf = tf.cast(adv_seq, tf.float32)
    returns_tf = tf.cast(returns_seq, tf.float32)
    h0_tf = tf.cast(rollout['h0'], tf.float32)
    c0_tf = tf.cast(rollout['c0'], tf.float32)

    # normalize advantage
    adv_tf = (adv_tf - tf.reduce_mean(adv_tf)) / (tf.math.reduce_std(adv_tf) + 1e-8)

    # Build dataset: each example is one environment's full T-sequence
    ds = tf.data.Dataset.from_tensor_slices(
        (ang_tf, lin_tf, rot_tf, laser_dist_tf, laser_type_tf,
         actions_tf, look_dir_tf, old_logp_tf, adv_tf, returns_tf, h0_tf, c0_tf)
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
             mb_actions, mb_look_dir, mb_old_logp, mb_adv, mb_returns, mb_h0, mb_c0) = batch

            # model_lock.acquire_write()
            # call compiled train step
            _train_step(mb_ang, mb_lin, mb_rot, mb_laser_dist, mb_laser_type,
                        mb_actions, mb_look_dir, mb_old_logp, mb_adv, mb_returns, mb_h0, mb_c0)

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


powers_of_two = 2 ** np.arange(NUM_ACTIONS - 2, -1, -1)


def sim_send_actions(conn, actions_per_player, look_directions):
    """
    Send actions back to sim. actions_per_player: numpy array shape (N, NUM_ACTIONS) 0/1
    """
    # Ensure it's integers
    arr = actions_per_player[:, :-1].astype(np.int8)

    # Compute powers of 2 for each bit (assuming most significant bit first)
    # powers_of_two = 2 ** np.arange(arr.shape[1] - 1, -1, -1)

    # Multiply and sum along axis 1
    pulse_input_set_per_player = arr.dot(powers_of_two)
    return send_model_outputs(conn, pulse_input_set_per_player, look_directions)


def collect_rollout(conn, rollout_length=SEQ_LEN):
    """
    Collect a rollout of length T for all N players currently in sim.
    Returns the 'rollout' dict used by ppo_update.
    """
    # initial read to get number of players N and initial states (sim returns numpy)
    T = rollout_length

    # TensorArray buffers (write(t, tensor) each step). We'll slice the stacks at the end.
    ang_buf = tf.TensorArray(tf.float32, size=T)
    lin_buf = tf.TensorArray(tf.float32, size=T)
    rot_buf = tf.TensorArray(tf.float32, size=T)
    laser_dist_buf = tf.TensorArray(tf.float32, size=T)
    laser_type_buf = tf.TensorArray(tf.int32, size=T)
    actions_buf = tf.TensorArray(tf.int32, size=T)  # store actions as int32
    look_dir_buf = tf.TensorArray(tf.float32, size=T)
    old_logp_buf = tf.TensorArray(tf.float32, size=T)
    values_buf = tf.TensorArray(tf.float32, size=T)
    rewards_buf = tf.TensorArray(tf.float32, size=T)
    dones_buf = tf.TensorArray(tf.float32, size=T)

    sim_state = sim_get_states(conn)
    N = int(sim_state['position'].shape[0])

    # initial LSTM states per env as TF tensors
    h = tf.zeros((N, LSTM_UNITS), dtype=tf.float32)
    c = tf.zeros((N, LSTM_UNITS), dtype=tf.float32)

    # also store initial states to use at training time (TF copies)
    h0 = tf.identity(h)
    c0 = tf.identity(c)

    t_written = 0

    for t in range(T):
        # t1 = round(time.time() * 1000)

        # convert sim_state (numpy) to TF tensors
        ang = tf.convert_to_tensor(sim_state['angvel'], dtype=tf.float32)  # (N,3)
        lin = tf.convert_to_tensor(sim_state['linvel'], dtype=tf.float32)  # (N,3)
        rot = tf.convert_to_tensor(sim_state['rotation'], dtype=tf.float32)  # (N,3)
        laser_dist = tf.convert_to_tensor(sim_state['laser']['distance'], dtype=tf.float32)  # (N, LASERS, 1)
        laser_type = tf.convert_to_tensor(sim_state['laser']['type'].astype(np.int32), dtype=tf.int32)  # (N, LASERS)

        # write states to TF buffers
        ang_buf = ang_buf.write(t, ang)
        lin_buf = lin_buf.write(t, lin)
        rot_buf = rot_buf.write(t, rot)
        laser_dist_buf = laser_dist_buf.write(t, laser_dist)
        laser_type_buf = laser_type_buf.write(t, laser_type)

        # run model step (TF accepts tf.Tensors)
        policy_tf, value_tf, look_tf, h_tf, c_tf = nav_model_step_tf(
            ang, lin, rot, laser_dist, laser_type, h, c
        )

        # sample actions in TF
        rnd = tf.random.uniform(tf.shape(policy_tf), dtype=policy_tf.dtype)
        actions_tf = tf.cast(rnd < policy_tf, tf.int32)  # shape (N, NUM_ACTIONS)

        # send actions to simulation (sim requires numpy)
        actions_np = actions_tf.numpy()
        look_np = look_tf.numpy()
        # t2 = round(time.time() * 1000)
        sim_send_actions(conn, actions_np, look_np)

        # update LSTM states from model outputs (TF tensors)
        h = tf.identity(h_tf)
        c = tf.identity(c_tf)

        # compute logp in TF (float)
        eps = 1e-8
        policy_float = tf.cast(policy_tf, tf.float32)
        actions_float = tf.cast(actions_tf, tf.float32)
        logp_tf = tf.reduce_sum(
            actions_float * tf.math.log(policy_float + eps) +
            (1.0 - actions_float) * tf.math.log(1.0 - policy_float + eps),
            axis=-1
        )  # shape (N,)

        # store action / logging info (TF)
        actions_buf = actions_buf.write(t, actions_tf)  # int32
        look_dir_buf = look_dir_buf.write(t, look_tf)
        old_logp_buf = old_logp_buf.write(t, logp_tf)  # float32
        values_buf = values_buf.write(t, tf.cast(value_tf, tf.float32))  # ensure float32

        # read next sim state (numpy) and convert to TF for storage
        sim_state = sim_get_states(conn)
        # t3 = round(time.time() * 1000)

        rewards = tf.convert_to_tensor(np.asarray(sim_state["reward"], dtype=np.float32), dtype=tf.float32)  # (N,)
        dones = tf.convert_to_tensor(np.asarray(sim_state["done"], dtype=np.float32), dtype=tf.float32)  # (N,)

        rewards_buf = rewards_buf.write(t, rewards)
        dones_buf = dones_buf.write(t, dones)

        # reset LSTM states where done==1 (TF ops)
        mask = 1.0 - tf.expand_dims(dones, axis=-1)  # shape (N,1)
        h = h * mask  # broadcasting: (N, LSTM_UNITS) * (N,1)
        c = c * mask

        for t in [ang_buf, lin_buf, rot_buf, laser_dist_buf, laser_type_buf, actions_buf, look_dir_buf, old_logp_buf,
                  values_buf, rewards_buf, dones_buf]:
            t.mark_used()

        # check living players (convert to python scalar for control flow)
        living_players = float(tf.reduce_sum(mask).numpy())
        t_written += 1

        if living_players == 0.0:
            if t_written == 1:
                return None
            break

        # t4 = round(time.time() * 1000)
        # sim_time = t3 - t2
        # total_time = t4 - t1
        # model_time = total_time - sim_time
        # print(f"total time: {total_time}ms, model time: {model_time}ms, simulation time: {sim_time}ms")

    # stack and slice to actual length t_written
    ang_buf = ang_buf.stack()[:t_written]  # (Tcollected, N, 3)
    lin_buf = lin_buf.stack()[:t_written]
    rot_buf = rot_buf.stack()[:t_written]
    laser_dist_buf = laser_dist_buf.stack()[:t_written]  # (Tcollected, N, LASERS, 1)
    laser_type_buf = laser_type_buf.stack()[:t_written]
    actions_buf = actions_buf.stack()[:t_written]  # (Tcollected, N, NUM_ACTIONS)
    look_dir_buf = look_dir_buf.stack()[:t_written]  # (Tcollected, N, NUM_ACTIONS)
    old_logp_buf = old_logp_buf.stack()[:t_written]  # (Tcollected, N)
    values_buf = values_buf.stack()[:t_written]  # (Tcollected, N)
    rewards_buf = rewards_buf.stack()[:t_written]  # (Tcollected, N)
    dones_buf = dones_buf.stack()[:t_written]  # (Tcollected, N)
    rollout = {
        'angvel': ang_buf,
        'linvel': lin_buf,
        'rot': rot_buf,
        'laser_dist': laser_dist_buf,
        'laser_type': laser_type_buf,
        'actions': actions_buf,
        'look_dir': look_dir_buf,
        'old_logp': old_logp_buf,
        'values': values_buf,
        'rewards': rewards_buf,
        'dones': dones_buf,
        'h0': h0,  # TF tensor shape (N, LSTM_UNITS)
        'c0': c0,
        'h_last': h,  # final TF tensors (N, LSTM_UNITS)
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
    print(
        f"[Info] LR will be divided by {LR_MULTIPLIER_PER_COLLAPSE_DETECTION} for {REQUIRED_STABLE_UPDATES_AFTER_COLLAPSE} steps")

    # model_lock.acquire_write()

    # Rollback to last viable version.
    # Restore the last saved model (CURRENT_MODEL_PATH)
    # If we're in a step that just precedes a saving, it means that the last saved model (CURRENT_MODEL_PATH) is also dead.
    # So we take the one before it (PREVIOUS_MODEL_PATH).
    nav_model.load_weights(CURRENT_MODEL_PATH if (current_update_step - 1) % BACKUP_RATE != 0 else PREVIOUS_MODEL_PATH)

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
            # while rollout_queue.qsize() > 0:
            #     rollout = rollout_queue.get_nowait()
            #     if not rollout:
            #         continue
            #
            #     for k, v in rollout.items():
            #         if k not in merged_rollout:
            #             merged_rollout[k] = v
            #         else:
            #             merged_rollout[k] = np.concatenate([merged_rollout[k], v],
            #                                                axis=0 if k in ["h0", "c0", "h_last", "c_last"] else 1)

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
        except KeyboardInterrupt:
            return
        except Exception as e:
            # print(e)
            save_model("model-error.keras")
            raise e
