import queue
import threading
import pandas as pd
from pathlib import Path
from typing import Sequence


class DataSaver:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._queue = queue.Queue()

        self._mapping: dict[str, pd.DataFrame] = {}

        self._thread = threading.Thread(target=self._run)
        self._running = True
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                filename, df = self._queue.get()
                self._mapping[filename] = pd.concat([self._mapping.get(filename, pd.DataFrame()), df], axis=1, inplace=True, ignore_index=True, sort=False)
            except queue.Empty:
                pass

    @staticmethod
    def complete():
        DataSaver._instance._running = False
        DataSaver._instance._queue.join()
        DataSaver._instance._thread.join()
    
    @staticmethod
    def save_to_local():
        for filename, df in DataSaver._instance._mapping.items():
            df.to_csv(filename.with_suffix('.csv'), index=False)
            df.to_parquet(filename.with_suffix('.parquet'), index=False)
        DataSaver._instance._mapping.clear()

    @staticmethod
    def save(filename: Path, name: str, vals: Sequence[float]):
        df = pd.DataFrame({name: vals})
        DataSaver._queue.put((filename, df))
