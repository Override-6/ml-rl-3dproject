import asyncio
import threading


class AsyncRWLock:
    def __init__(self):
        self._readers = 0
        self._writer = threading.Lock()
        self._readers_lock = threading.Lock()

    def acquire_read(self):
        with self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                # first reader locks the writer
                self._writer.acquire()

    def release_read(self):
        with self._readers_lock:
            self._readers -= 1
            if self._readers == 0:
                # last reader releases the writer
                self._writer.release()

    def acquire_write(self):
        self._writer.acquire()

    def release_write(self):
        self._writer.release()