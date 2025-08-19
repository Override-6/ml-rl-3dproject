import os
import warnings

import numpy as np

from data import PlayerState, Input
from src.hyperparameters import FEATURE_DIM, LSTM_UNITS, LASERS_PER_PLAYER, NB_COMPONENT_TYPES, NUM_ACTIONS
import tensorflow as tf
from tensorflow.keras import layers

# ------------------------
# Build model class
# ------------------------
class NavigationModel(tf.keras.Model):
    def __init__(self, file: str | None = None):
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

        if file and os.path.exists(file):
            self.load_weights(file)
        else:
            warnings.warn(f"File {file} not found, using random weights...")

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
        self.build((batch_size, 1, 1))

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
