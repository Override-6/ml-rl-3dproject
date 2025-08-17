import asyncio


class AsyncRWLock:
    def __init__(self):
        self._readers = 0
        self._writer = asyncio.Lock()
        self._readers_lock = asyncio.Lock()

    async def acquire_read(self):
        async with self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                # first reader locks the writer
                await self._writer.acquire()

    async def release_read(self):
        async with self._readers_lock:
            self._readers -= 1
            if self._readers == 0:
                # last reader releases the writer
                self._writer.release()

    async def acquire_write(self):
        await self._writer.acquire()

    def release_write(self):
        self._writer.release()