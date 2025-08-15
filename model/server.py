import socket
import struct

import model
from data import Vec3, LaserHit, PlayerState

PLAYER_LASER_COUNT = 4  # adjust to match your Rust constant


class InputSet:
    def __init__(self, forward: float = 0.0, turn: float = 0.0):
        self.forward = forward
        self.turn = turn

    def to_bytes(self) -> bytes:
        # Assuming 2 floats per InputSet
        return struct.pack('<ff', self.forward, self.turn)

def recv_exact(conn, size):
    buf = b''
    while len(buf) < size:
        chunk = conn.recv(size - len(buf))
        if not chunk:
            raise ConnectionError("Client disconnected")
        buf += chunk
    return buf

def read_player_states(conn):
    # Read player count (u32)
    raw_len = recv_exact(conn, 4)
    player_count = struct.unpack('<I', raw_len)[0]

    # Read PlayerStates
    player_states = []
    player_state_size = 96 # sizeof(PlayerState);

    data = recv_exact(conn, player_count * player_state_size)

    for i in range(player_count):
        offset = i * player_state_size
        chunk = data[offset:offset + player_state_size]

        reward = struct.unpack('<f', chunk[0:4])[0]
        done = struct.unpack('<i', chunk[4:8])[0] == 1

        print(done)

        # Unpack Vec3s
        vecs = struct.unpack('<' + 'f' * 12, chunk[8:48 + 8])
        pos = Vec3(*vecs[0:3])
        ang = Vec3(*vecs[3:6])
        lin = Vec3(*vecs[6:9])
        rot = Vec3(*vecs[9:12])

        # Unpack lasers with padding
        lasers = []
        for j in range(PLAYER_LASER_COUNT):
            base = 48 + 8 + j * 8
            distance = struct.unpack('<f', chunk[base:base + 4])[0]
            component_type = chunk[base + 4]  # u8
            lasers.append(LaserHit(distance=distance, component_type=component_type))

        player_states.append(PlayerState(reward, done, pos, ang, lin, rot, lasers))

    print("Received Player States")
    return player_states


def send_model_outputs(conn, outputs):
    conn.sendall(struct.pack('<I', len(outputs)))  # send count
    for inp in outputs:
        conn.sendall(struct.pack('B', inp))

    print("Sent model outputs")


def handle_simulation_loop(conn):
    with conn:
        while True:
            player_states = read_player_states(conn)

            outputs = model.step(player_states)

            send_model_outputs(conn, outputs)


def main():
    HOST = 'localhost'
    PORT = 9999
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Model server listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            print(f"Connected by {addr}")
            try:
                handle_simulation_loop(conn)
            except Exception as e:
                print(e)

if __name__ == "__main__":
    main()
