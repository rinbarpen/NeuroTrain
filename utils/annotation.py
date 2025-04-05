import logging
import time

def doc(desc: str):
    def decorator(f):
        f.__doc__ = desc
        return f
    return decorator

def time_cost(f):
    def wrapper(*args, **kwargs):
        begin = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        logging.debug(f"Function {f.__name__} took {end - begin} ns")
        return result
    return wrapper

class timer:
    def __init__(self, name="block"):
        self.name = name

    def __enter__(self):
        self.begin = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        print(f"{self.name} took {end - self.begin} s")
