from pydantic import Field, BaseModel
import time
import numpy as np

class TimerUnit(BaseModel):
    start_time: float = Field(float('NAN'))
    end_time: float = Field(float('NAN'))

class Timer:
    def __init__(self):
        self.time_map: dict[str, TimerUnit] = {}

    def start(self, task: str=""):
        unit = TimerUnit()
        unit.start_time = time.time()
        self.time_map[task] = unit

    def stop(self, task: str=""):
        try:
            self.time_map[task].end_time = time.time()
        except Exception:
            pass
    def elapsed_time(self, task: str=""):
        try:
            return self.time_map[task].end_time - self.time_map[task].start_time
        except Exception:
            return float('NAN')
    def all_elapsed_time(self):
        costs = {}
        for task in self.time_map.keys():
            costs[task] = self.elapsed_time(task)
        return costs
    def total_elapsed_time(self) -> float:
        total_cost = np.array(self.all_elapsed_time().values()).sum()
        return total_cost
