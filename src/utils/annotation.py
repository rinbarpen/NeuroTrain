import logging
import time
from functools import wraps

def deprecated(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        print(f"Function {f.__name__} is deprecated")
        return f(*args, **kwargs)
    return wrapper


def buildin(desc: str=""):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            print(f"Using buildin function {f.__name__}")
            return f(*args, **kwargs)
        return wrapper
    return decorator

def time_cost(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        begin = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f"Function {f.__name__} took {end - begin} ns")
        return result
    return wrapper

class timer:
    def __init__(self, name="block"):
        self.name = name
        self.begin = None

    def __enter__(self):
        self.begin = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        print(f"{self.name} took {end - self.begin} s")
