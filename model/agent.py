# Requires: tensorflow >= 2.x
import os
import queue
import threading

import numpy as np

from data import Input
from rw_lock import AsyncRWLock

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from tensorflow.keras import layers

from simulation import read_player_states, send_model_outputs

import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"

# ------------------------
# Hyperparameters
# ------------------------
LASERS_PER_PLAYER = 5
NB_COMPONENT_TYPES = 5
NUM_ACTIONS = len(Input)  # number of binary action outputs
FEATURE_DIM = 128
LSTM_UNITS = 64
SEQ_LEN = 20 * 5  # rollout horizon (timesteps collected before update). is simulation's hz times 5 seconds
PPO_EPOCHS = 4
MINIBATCH_SIZE = 64
CLIP_EPS = 0.2
VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.05
LR = 3e-4
GAMMA = 0.95
LAMBDA = 0.95


# ------------------------
# Build model class
# ------------------------
class NavigationModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # feature extractor: per-timestep inputs -> feature vector
        # We'll implement it as layers we can call manually
        # Note: not using functional Model() because we need flexibility with initial states
        self.pos_layer = layers.InputLayer(input_shape=(3,))
        self.angvel_layer = layers.InputLayer(input_shape=(3,))
        self.linvel_layer = layers.InputLayer(input_shape=(3,))
        self.rot_layer = layers.InputLayer(input_shape=(3,))

        # lasers
        self.laser_dist_layer = layers.InputLayer(input_shape=(LASERS_PER_PLAYER, 1))
        self.laser_type_layer = layers.InputLayer(input_shape=(LASERS_PER_PLAYER,))

        self.type_emb = layers.Embedding(input_dim=NB_COMPONENT_TYPES, output_dim=4, name="type_emb")
        self.conv1 = layers.Conv1D(16, 3, activation='relu', padding='same')
        self.conv2 = layers.Conv1D(32, 3, activation='relu', padding='same')
        self.flatten = layers.Flatten()

        # dense feature head
        self.dense1 = layers.Dense(FEATURE_DIM, activation='relu')
        self.dense2 = layers.Dense(FEATURE_DIM, activation='relu')

        # LSTM layer (can accept initial_state param on call)
        self.lstm = layers.LSTM(LSTM_UNITS, return_state=True, return_sequences=True, name="lstm")

        # Heads
        # Policy uses sigmoid for independent binary actions
        self.policy_head = layers.Dense(NUM_ACTIONS, activation='sigmoid', name="policy")
        self.value_head = layers.Dense(1, activation=None, name="value")

        self._dummy_build()

    def copy_from(self, other: "NavigationModel"):
        """
        Copy all weights from another NavigationModel into self.
        """
        if not isinstance(other, NavigationModel):
            raise ValueError("Can only copy from another NavigationModel instance.")

        assert self is not other

        # Iterate over all layers and set weights
        for self_layer, other_layer in zip(self.layers, other.layers):
            self_layer.set_weights(other_layer.get_weights())

    def _dummy_build(self):
        batch_size = 1
        dummy_pos = tf.zeros((batch_size, 3), dtype=tf.float32)
        dummy_angvel = tf.zeros((batch_size, 3), dtype=tf.float32)
        dummy_linvel = tf.zeros((batch_size, 3), dtype=tf.float32)
        dummy_rot = tf.zeros((batch_size, 3), dtype=tf.float32)
        dummy_laser_dist = tf.zeros((batch_size, LASERS_PER_PLAYER, 1), dtype=tf.float32)
        dummy_laser_type = tf.zeros((batch_size, LASERS_PER_PLAYER), dtype=tf.int32)
        dummy_h = tf.zeros((batch_size, LSTM_UNITS), dtype=tf.float32)
        dummy_c = tf.zeros((batch_size, LSTM_UNITS), dtype=tf.float32)

        # call step once to build all layers/variables
        _ = self.step(
            dummy_pos, dummy_angvel, dummy_linvel, dummy_rot,
            dummy_laser_dist, dummy_laser_type, dummy_h, dummy_c
        )

    def copy(self):
        new_model = NavigationModel()
        new_model.copy_from(self)
        return new_model

    def extract_features(self, pos, angvel, linvel, rot, laser_dist, laser_type):
        """Extract single-timestep features. Inputs are tensors with shape (batch, ...)"""
        # apply input layers so shapes are checked
        # _ = self.pos_layer(pos)
        # _ = self.angvel_layer(angvel)
        # _ = self.linvel_layer(linvel)
        # _ = self.rot_layer(rot)
        # _ = self.laser_dist_layer(laser_dist)
        # _ = self.laser_type_layer(laser_type)

        emb = self.type_emb(laser_type)  # (batch, LASERS, emb)
        laser_features = tf.concat([laser_dist, emb], axis=-1)  # (batch, LASERS, 1+emb)
        x = self.conv1(laser_features)
        x = self.conv2(x)
        x = self.flatten(x)  # (batch, conv_features)

        state_feat = tf.concat([pos, angvel, linvel, rot, x], axis=-1)
        state_feat = self.dense1(state_feat)
        state_feat = self.dense2(state_feat)  # (batch, FEATURE_DIM)
        return state_feat

    @tf.function
    def step(self, pos, angvel, linvel, rot, laser_dist, laser_type, h, c):
        """
        Single-step forward with initial LSTM state (h,c).
        Inputs shapes: each (batch, ...)
        h, c shapes: (batch, LSTM_UNITS)
        Returns:
            policy: (batch, NUM_ACTIONS)            # probabilities
            value: (batch,)                        # scalar value per env
            h_next: (batch, LSTM_UNITS)
            c_next: (batch, LSTM_UNITS)
        """
        feat = self.extract_features(pos, angvel, linvel, rot, laser_dist, laser_type)
        feat_seq = tf.expand_dims(feat, axis=1)  # (batch, 1, FEATURE_DIM)

        # LSTM returns (batch, 1, units), h_next (batch,units), c_next (batch,units)
        out_seq, h_next, c_next = self.lstm(feat_seq, initial_state=[h, c])

        # squeeze time dim -> (batch, units)
        out = tf.squeeze(out_seq, axis=1)

        policy = self.policy_head(out)  # (batch, NUM_ACTIONS)
        value = tf.squeeze(self.value_head(out), axis=-1)  # (batch,)
        return policy, value, h_next, c_next

    @tf.function
    def forward_sequence(self, pos_seq, angvel_seq, linvel_seq, rot_seq, laser_dist_seq, laser_type_seq,
                         h0=None, c0=None):
        """
        Forward a whole sequence for training.
        Each input shape: (batch, seq_len, ...)
        Optionally provide initial states (batch, LSTM_UNITS) or defaults to zeros.
        Returns:
            policy_seq: (batch, seq_len, NUM_ACTIONS)
            values_seq: (batch, seq_len)
            h_last: (batch, LSTM_UNITS)
            c_last: (batch, LSTM_UNITS)
        """
        batch = tf.shape(pos_seq)[0]
        seq_len = tf.shape(pos_seq)[1]

        # TimeDistributed extraction: apply extract_features across time
        # We will reshape to merge batch and time, run extract_features, then reshape back
        def merge_bt(x):  # (batch, seq_len, feat) -> (batch*seq_len, feat)
            s = tf.shape(x)
            return tf.reshape(x, (s[0] * s[1],) + tuple(x.shape.as_list()[2:]))

        p = merge_bt(pos_seq)
        a = merge_bt(angvel_seq)
        l = merge_bt(linvel_seq)
        r = merge_bt(rot_seq)
        ld = merge_bt(laser_dist_seq)
        lt = merge_bt(laser_type_seq)

        feat_bt = self.extract_features(p, a, l, r, ld, lt)  # (batch*seq_len, FEATURE_DIM)

        # Restore (batch, seq_len, FEATURE_DIM)
        feat_seq = tf.reshape(feat_bt, (batch, seq_len, FEATURE_DIM))

        if h0 is None:
            h0 = tf.zeros((batch, LSTM_UNITS))
            c0 = tf.zeros((batch, LSTM_UNITS))

        # Run LSTM across the whole sequence
        all_outs_and_states = self.lstm(feat_seq, initial_state=[h0, c0])
        outputs = all_outs_and_states[0]  # (batch, seq_len, LSTM_UNITS)
        h_last = all_outs_and_states[1]
        c_last = all_outs_and_states[2]

        policy_seq = self.policy_head(outputs)  # (batch, seq_len, NUM_ACTIONS)
        values_seq = tf.squeeze(self.value_head(outputs), axis=-1)  # (batch, seq_len)
        return policy_seq, values_seq, h_last, c_last


# instantiate train and prediction model
predict_model = NavigationModel()
train_model = NavigationModel()

model_update_lock = threading.Lock()


def update_prediction_model():
    model_update_lock.acquire()
    train_model.copy_from(predict_model)
    model_update_lock.release()

def copy_train_model():
    model_update_lock.acquire()
    result = train_model.copy()
    model_update_lock.release()
    return result

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
def ppo_update(rollout, optimizer):
    """
    rollout: dict with keys:
      pos: (T, N, 3)
      angvel, linvel, rot: same as pos
      laser_dist: (T, N, LASERS_PER_PLAYER, 1)
      laser_type: (T, N, LASERS_PER_PLAYER)
      actions: (T, N, SEQ_LEN, NUM_ACTIONS)   # sequence of actions per environment step
      old_logp: (T, N, SEQ_LEN)
      values: (T, N, SEQ_LEN)
      rewards: (T, N)
      dones: (T, N) 0/1
      h0: (N, LSTM_UNITS) initial states at rollout start
      c0: (N, LSTM_UNITS)
    nav_model: model instance
    """
    T, N, _ = rollout['actions'].shape

    # compute last_value for bootstrap using nav_model (use last observation)
    last_pos = rollout['pos'][-1]
    last_angvel = rollout['angvel'][-1]
    last_linvel = rollout['linvel'][-1]
    last_rot = rollout['rot'][-1]
    last_laser_dist = rollout['laser_dist'][-1]
    last_laser_type = rollout['laser_type'][-1]
    h_last = rollout['h_last']  # states after final step, shape (N, LSTM_UNITS)
    c_last = rollout['c_last']

    last_policy, last_value, _, _ = train_model.step(
        last_pos, last_angvel, last_linvel, last_rot, last_laser_dist, last_laser_type, h_last, c_last
    )
    last_value = last_value.numpy()  # (N,)

    # compute advantages & returns
    advantages, returns = compute_gae(
        rollout['rewards'], rollout['values'], rollout['dones'], last_value, gamma=GAMMA, lam=LAMBDA
    )

    # Prepare training data: we will train across epochs using minibatches
    pos_seq = np.transpose(rollout['pos'], (1, 0, 2))  # (N, T, 3)
    ang_seq = np.transpose(rollout['angvel'], (1, 0, 2))
    lin_seq = np.transpose(rollout['linvel'], (1, 0, 2))
    rot_seq = np.transpose(rollout['rot'], (1, 0, 2))
    laser_dist_seq = np.transpose(rollout['laser_dist'], (1, 0, 2, 3))  # (N,T, LASERS,1)
    laser_type_seq = np.transpose(rollout['laser_type'], (1, 0, 2))
    actions_seq = np.transpose(rollout['actions'], (1, 0, 2))
    old_logp_seq = np.transpose(rollout['old_logp'], (1, 0))
    adv_seq = np.transpose(advantages, (1, 0))
    returns_seq = np.transpose(returns, (1, 0))
    # values_seq = np.transpose(rollout['values'], (1, 0))

    # We'll train for multiple epochs and sample minibatches of environments
    num_samples = N
    idxs = np.arange(num_samples)

    for epoch in range(PPO_EPOCHS):
        np.random.shuffle(idxs)
        print("Epoch {}".format(epoch + 1), end="\r")
        for start in range(0, num_samples, MINIBATCH_SIZE):
            mb_idxs = idxs[start:start + MINIBATCH_SIZE]
            # minibatch sequences (batch = mb_size)
            mb_pos = tf.convert_to_tensor(pos_seq[mb_idxs], dtype=tf.float32)  # (mb, T, 3)
            mb_ang = tf.convert_to_tensor(ang_seq[mb_idxs], dtype=tf.float32)
            mb_lin = tf.convert_to_tensor(lin_seq[mb_idxs], dtype=tf.float32)
            mb_rot = tf.convert_to_tensor(rot_seq[mb_idxs], dtype=tf.float32)
            mb_laser_dist = tf.convert_to_tensor(laser_dist_seq[mb_idxs], dtype=tf.float32)
            mb_laser_type = tf.convert_to_tensor(laser_type_seq[mb_idxs], dtype=tf.int32)
            mb_actions = tf.convert_to_tensor(actions_seq[mb_idxs], dtype=tf.float32)  # (mb, T, NUM_ACTIONS)
            mb_old_logp = tf.convert_to_tensor(old_logp_seq[mb_idxs], dtype=tf.float32)  # (mb, T)
            mb_adv = tf.convert_to_tensor(adv_seq[mb_idxs], dtype=tf.float32)  # (mb, T)
            mb_returns = tf.convert_to_tensor(returns_seq[mb_idxs], dtype=tf.float32)

            # Use initial LSTM states at rollout start for these envs
            mb_h0 = tf.convert_to_tensor(rollout['h0'][mb_idxs], dtype=tf.float32)
            mb_c0 = tf.convert_to_tensor(rollout['c0'][mb_idxs], dtype=tf.float32)

            # Forward pass: get policy_seq and values_seq from model
            with tf.GradientTape() as tape:
                policy_seq, values_seq_pred, _, _ = train_model.forward_sequence(
                    mb_pos, mb_ang, mb_lin, mb_rot, mb_laser_dist, mb_laser_type, h0=mb_h0, c0=mb_c0
                )
                # shapes: policy_seq (mb, T, NUM_ACTIONS), values_seq_pred (mb, T)

                # compute log_probs under current policy
                # flatten time+batch for convenience
                probs_flat = tf.reshape(policy_seq, (-1, NUM_ACTIONS))  # (mb*T, NUM_ACTIONS)
                actions_flat = tf.reshape(mb_actions, (-1, NUM_ACTIONS))
                logp_flat = bernoulli_log_probs(probs_flat, actions_flat)  # (mb*T,)
                logp = tf.reshape(logp_flat, (tf.shape(mb_actions)[0], tf.shape(mb_actions)[1]))  # (mb, T)

                # likewise flatten predicted values
                values_flat = tf.reshape(values_seq_pred, (-1,))
                values_pred = tf.reshape(values_flat, (tf.shape(mb_actions)[0], tf.shape(mb_actions)[1]))  # (mb, T)

                # ratio = exp(new_logp - old_logp)
                ratio = tf.exp(logp - mb_old_logp)

                # advantages & returns as tensors
                mb_adv_ = mb_adv
                mb_ret_ = mb_returns

                # policy loss with clipping (sum over time, mean over minibatch)
                unclipped = ratio * mb_adv_
                clipped = tf.clip_by_value(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv_
                policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

                # value loss (MSE)
                value_loss = VALUE_COEFF * tf.reduce_mean(tf.square(mb_ret_ - values_pred))

                # entropy (encourage exploration) average over actions and time
                eps = 1e-8
                entropy_per_action = -(policy_seq * tf.math.log(policy_seq + eps) + (1 - policy_seq) * tf.math.log(
                    1 - policy_seq + eps))
                entropy = tf.reduce_mean(entropy_per_action)
                entropy_loss = -ENTROPY_COEFF * entropy

                total_loss = policy_loss + value_loss + entropy_loss

            grads = tape.gradient(total_loss, train_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, train_model.trainable_variables))

    update_prediction_model()
    # done with update
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


# Example rollout collection loop skeleton
def collect_rollout(conn, rollout_length=SEQ_LEN):
    """
    Collect a rollout of length T for all N players currently in sim.
    Returns the 'rollout' dict used by ppo_update.
    """
    prediction_model = copy_train_model()
    # initial read to get number of players N and initial states
    sim_state = sim_get_states(conn)
    N = sim_state['position'].shape[0]
    T = rollout_length

    # buffers (T, N, ...)
    pos_buf = np.zeros((T, N, 3), dtype=np.float32)
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
        pos = sim_state['position'].astype(np.float32)
        ang = sim_state['angvel'].astype(np.float32)
        lin = sim_state['linvel'].astype(np.float32)
        rot = sim_state['rotation'].astype(np.float32)
        laser_dist = sim_state['laser']['distance'].astype(np.float32)
        laser_type = sim_state['laser']['type'].astype(np.int32)

        # store state
        pos_buf[t] = pos
        ang_buf[t] = ang
        lin_buf[t] = lin
        rot_buf[t] = rot
        laser_dist_buf[t] = laser_dist
        laser_type_buf[t] = laser_type

        # run model step (batch)
        policy_tf, value_tf, h_tf, c_tf = prediction_model.step(
            tf.convert_to_tensor(pos), tf.convert_to_tensor(ang),
            tf.convert_to_tensor(lin), tf.convert_to_tensor(rot),
            tf.convert_to_tensor(laser_dist), tf.convert_to_tensor(laser_type),
            tf.convert_to_tensor(h), tf.convert_to_tensor(c)
        )
        policy = policy_tf.numpy()  # (N, NUM_ACTIONS)
        value = value_tf.numpy()
        h = h_tf.numpy()
        c = c_tf.numpy()

        # Sample actions from bernoulli (we use probabilities policy)
        actions = (np.random.rand(*policy.shape) < policy).astype(np.float32)  # (N, SEQ_LEN, NUM_ACTIONS,)
        # compute log_probs under the policy we used (store old_logp for PPO)
        eps = 1e-8
        logp = np.sum(actions * np.log(policy + eps) + (1 - actions) * np.log(1 - policy + eps),
                      axis=-1)  # (N, SEQ_LEN)

        # send actions to simulation
        sim_send_actions(conn, actions)

        # after sim step, read rewards and done flags for each player
        # assuming you can call sim_get_rewards() or sim provides them in next sim_get_states()
        # For simplicity, we call sim_get_states again to read reward/done or make a dedicated call
        # Replace the following with your actual reward/done retrieval
        sim_state = sim_get_states(conn)  # compute from the next state the rewards, dones etc
        # Here you must obtain rewards and done for each player. Replace these lines:
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
        'pos': pos_buf,
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


def agent_loop(rollout_queue: queue.Queue):
    try:
        num_updates = 10000
        for update in range(num_updates):
            # merge all rollouts
            merged_rollout = rollout_queue.get() # at least one rollout
            while rollout_queue.qsize() > 0:
                rollout = rollout_queue.get_nowait()
                if not rollout:
                    continue
                for k, v in rollout.items():
                    if k not in merged_rollout:
                        merged_rollout[k] = v
                    else:
                        merged_rollout[k] = np.concatenate([merged_rollout[k], v], axis=0 if k in ["h0", "c0", "h_last", "c_last"] else 1)

            print("training and Updating model... N=", merged_rollout["h_last"].shape[0])
            # train on rollout
            ppo_update(merged_rollout, optimizer)
            # logging / saving model checkpoints etc.
            print(f"Finished update {update}")
    except Exception as e:
        print(e)
        raise e