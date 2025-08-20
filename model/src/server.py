import queue
import socket
import threading
from threading import Thread

from src import agent
from simulation import send_reset

rollout_queue = queue.Queue()

def handle_client_thread(conn, rollout_queue: queue.Queue):
    """Dedicated thread for each client using blocking I/O."""

    addr = conn.getpeername()
    print(f"[Thread] Connected with {addr}")

    try:
        while True:
            # Blocking I/O: use agent.collect_rollout_blocking
            # You need a version of your agent that works with sockets directly
            rollout = agent.collect_rollout(conn)
            send_reset(conn)
            # Use asyncio.run_coroutine_threadsafe to put into asyncio queue
            rollout_queue.put_nowait(rollout)
    except Exception as e:
        print(f"[Thread] Error: {e}")
        raise
    finally:
        conn.close()
        print(f"[Thread] Disconnected {addr}")

def main():
    HOST = 'localhost'
    PORT = 9999

    # Start agent loop
    Thread(target=agent.agent_loop, args=[rollout_queue], daemon=True).start()

    # Create server socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen()
    print(f"Threaded server listening on {HOST}:{PORT}")

    while True:
        client_sock, _ = server_sock.accept()
        thread = threading.Thread(target=handle_client_thread, args=(client_sock, rollout_queue), daemon=True)
        thread.start()

if __name__ == "__main__":
    main()
