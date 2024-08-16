import time
from threading import Thread, Event

class FrameWait:
    def __init__(self, fps = 30.):
        self.fps = fps
        self._thread = Thread(target = self._wait_thread, daemon=True)
        self._frame_start = Event()
        self._frame_end = Event()


    def _wait_thread(self):
        while True:
            self._frame_start.wait()
            self._frame_start.clear()
            time.sleep(1/self.fps)
            self._frame_end.set()


    def start_frame(self):
        if not self._thread.is_alive():
            self._thread.start()
        self._frame_end.clear()
        self._frame_start.set()

    def wait_frame(self):
        self._frame_end.wait()

