import asyncio
import threading

import agent

rollout_queue = asyncio.Queue()

def handle_client_thread(reader, writer):
    """Run in a dedicated thread for each client."""
    async def client_loop():
        addr = writer.get_extra_info('peername')
        print(f"[Thread] Connected with {addr}")
        try:
            while True:
                rollout = await agent.collect_rollout(reader, writer)
                await rollout_queue.put(rollout)
        except Exception as e:
            print(f"[Thread] Error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    # Run the async loop in this thread
    asyncio.run(client_loop())

async def handle_client(reader, writer):
    # Spawn a thread for this client
    thread = threading.Thread(target=handle_client_thread, args=(reader, writer))
    thread.start()


async def main():
    HOST = 'localhost'
    PORT = 9999
    asyncio.create_task(agent.agent_loop(rollout_queue))

    server = await asyncio.start_server(handle_client, HOST, PORT)
    async with server:
        print(f"Model server listening on {HOST}:{PORT}")
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
