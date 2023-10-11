import datetime 
import functools 
import os 
import pathlib 
from typing import Tuple 

SOURCE_DIRECTORY: os.PathLike = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIRECTORY: os.PathLike = os.path.dirname(SOURCE_DIRECTORY)

def get_now_str() -> str:
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

@functools.lru_cache
def get_project_subdirectory(name: str) -> os.PathLike: 
    absolute_path: os.PathLike = os.path.join(PROJECT_DIRECTORY, name)
    if os.path.exists(absolute_path) is False: 
        os.mkdir(absolute_path)
    return absolute_path

LOG_DIRECTORY: os.PathLike = get_project_subdirectory("logs")

def human_bytes_str(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB")
    power = 2**10

    for unit in units:
        if num_bytes < power:
            return f"{num_bytes:.1f} {unit}"

        num_bytes /= power

    return f"{num_bytes:.1f} TB"

def human_seconds_str(seconds: int) -> str:
    units: Tuple[str] = ("seconds", "milliseconds", "microseconds")
    power: int = 1

    for unit in units:
        if seconds > power:
            return f"{seconds:.1f} {unit}"

        seconds *= 1000

    return f"{int(seconds)} nanoseconds"
