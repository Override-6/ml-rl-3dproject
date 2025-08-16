import socket

from agent import train_loop


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
                train_loop(conn)
            except Exception as e:
                raise

if __name__ == "__main__":
    main()
