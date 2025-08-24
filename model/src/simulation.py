import socket
import struct

import numpy as np

from src.hyperparameters import LASERS_PER_PLAYER


def recv_exact(conn: socket.socket, size):
    buf = b''
    while len(buf) < size:
        chunk = conn.recv(size - len(buf))
        if not chunk:
            raise ConnectionError("Client disconnected")
        buf += chunk
    return buf


def read_player_states(conn):
    # Read player count (u32)
    # print("Reading player states...")
    raw_len = recv_exact(conn, 4)
    N = struct.unpack('<I', raw_len)[0]

    # Read PlayerStates
    # player_state_size = 96  # sizeof(PlayerState);
    player_state_size = 64

    data = recv_exact(conn, N * player_state_size)

    state = {
        "reward": np.zeros(shape=(N,), dtype=np.float32),
        "done": np.zeros(shape=(N,), dtype=np.uint8),
        "action": np.zeros(shape=(N,), dtype=np.int32),
        "position": np.zeros(shape=(N, 3), dtype=np.float32),
        "rotation": np.zeros(shape=(N, 3), dtype=np.float32),
        "angvel": np.zeros(shape=(N, 3), dtype=np.float32),
        "linvel": np.zeros(shape=(N, 3), dtype=np.float32),
        "laser": {
            "distance": np.zeros(shape=(N, LASERS_PER_PLAYER, 1), dtype=np.float32),
            "type": np.zeros(shape=(N, LASERS_PER_PLAYER,), dtype=np.float32),
        }
    }

    for i in range(N):
        offset = i * player_state_size
        chunk = data[offset:offset + player_state_size]

        state["reward"][i] = struct.unpack('<f', chunk[0:4])[0]
        state["done"][i] = chunk[4] == 1

        # Unpack Vec3s
        vecs = struct.unpack('<' + 'f' * 12, chunk[8:48 + 8])
        state["position"][i] = np.array(vecs[0:3])
        state["rotation"][i] = np.array(vecs[3:6])
        state["angvel"][i] = np.array(vecs[6:9])
        state["linvel"][i] = np.array(vecs[9:12])

        # Unpack lasers with padding
        for j in range(LASERS_PER_PLAYER):
            base = 48 + 8 + j * 8
            distance = struct.unpack('<f', chunk[base:base + 4])[0]
            component_type = chunk[base + 4]  # u8
            state["laser"]["distance"][i, j] = distance
            state["laser"]["type"][i, j] = component_type

    # print("Received Player States")
    return state

def send_model_outputs(conn, pulse_outputs, directional_laser):
    conn.sendall(struct.pack('<I', 1)) # packet type: send_model_output
    conn.sendall(struct.pack('<I', len(pulse_outputs)))  # send count
    for i in pulse_outputs:
        conn.sendall(struct.pack('B', i))
    for i in directional_laser:
        conn.sendall(struct.pack('<f', i))

def send_reset(conn):
    conn.sendall(struct.pack('<I', 0)) # packet type: reset simulation
