from contextlib import contextmanager
from pydantic import Field, BaseModel
import time
import numpy as np
from typing import Literal

Precision = Literal["ms", "s"]

class TimerUnit(BaseModel):
    start_time: float = Field(float('NAN'))  # ms
    end_time: float = Field(float('NAN')) # ms

    def elapsed_time(self):
        return self.end_time - self.start_time

class Timer:
    def __init__(self, precision: Precision="ms"):
        self.time_map: dict[str, TimerUnit] = {}
        self._current_task: str = "undefined"
        self._current_unit: TimerUnit = TimerUnit()
        self.precision = precision

    def get_current_time(self) -> float:
        if self.precision == "ms":
            return time.time() * 1000
        return time.time()

    def start(self, task: str = "undefined"):
        self._current_task = task
        self._current_unit = TimerUnit(start_time=self.get_current_time())

    def stop(self):
        self._current_unit.end_time = self.get_current_time()
        self.time_map[self._current_task] = self._current_unit

    def elapsed_time(self, task: str = "undefined"):
        try:
            return self.time_map[task].elapsed_time()
        except Exception:
            return float('NAN')
    def all_elapsed_time(self):
        costs = {task: time_unit.elapsed_time() for task, time_unit in self.time_map.items()}
        return costs
    def total_elapsed_time(self) -> np.float64:
        total_cost = np.fromiter(self.all_elapsed_time().values(), dtype=np.float64).sum()
        return total_cost

    def reset(self):
        self.time_map.clear()
        self._current_task = "undefined"
        self._current_unit = TimerUnit()

    @contextmanager
    def timeit(self, task: str = "undefined"):
        try:
            self.start(task)
            yield
        finally:
            self.stop()
