import logging
import time
from functools import wraps

def singleton(cls):
    """
    单例模式装饰器，确保一个类只有一个实例。
    """
    instances = {}
    
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def deprecated(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        print(f"Function {f.__name__} is deprecated")
        return f(*args, **kwargs)
    return wrapper

def supported_in_future(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        print(f"Function {f.__name__} will be supported in the future.")
        return f(*args, **kwargs)
    return wrapper


@singleton
class Register:
    _modules: dict[str, set[str]] = {}

    @staticmethod
    def reg(module: str, name: str):
        try:
            Register._modules[module].append(name)
        except:
            Register._modules[module] = set(name)
    @staticmethod
    def include(module: str, name: str):
        return name in Register._modules[module]

def reg(module: str, name: str|None=None):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            Register.reg(name or f.__name__)
        return wrapper
    return decorator


def retry(times=3, delay=5):
    """
    A decorator for retrying a function if it raises an exception.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i < times - 1:
                        print(f"Function {func.__name__}: Attempt {i+1}/{times} failed with error: {e}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        print(f"Function {func.__name__}: Attempt {i+1}/{times} failed. No more retries.")
                        raise
        return wrapper
    return decorator


def buildin(desc: str=""):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # print(f"Using buildin function {f.__name__}")
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
