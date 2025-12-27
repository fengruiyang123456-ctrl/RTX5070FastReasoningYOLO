import time
from typing import Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class Timer:
    def __init__(self, use_cuda: bool = False) -> None:
        self.use_cuda = use_cuda and torch is not None and torch.cuda.is_available()
        self._start: Optional[float] = None
        self._end: Optional[float] = None
        self._start_event = None
        self._end_event = None
        if self.use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        if self.use_cuda and self._start_event is not None:
            self._start_event.record()
        else:
            self._start = time.perf_counter()

    def stop(self) -> float:
        if self.use_cuda and self._end_event is not None:
            self._end_event.record()
            torch.cuda.synchronize()
            return float(self._start_event.elapsed_time(self._end_event))
        if self._start is None:
            return 0.0
        self._end = time.perf_counter()
        return (self._end - self._start) * 1000.0
