"""
Miscellaneous utility functions
"""
import time

import py3nvml


def flatten_list(input_list: list) -> list:
    """
    Flattens a nested list that contains lists as its elements.
    Only goes one level deep (i.e. works on lists of lists but not lists of lists of lists).
    """
    return [item for sublist in input_list for item in sublist]


def get_available_gpus_devices(wait: bool = False, refresh_time: int = 10) -> list:
    """get a list of available gpus

    Args:
        wait (bool, optional): Wait until a gpu is free. Defaults to False.
        refresh_time (int, optional): how often to recheck if a gpu is available (only when wait=True). Defaults to 10.

    Raises:
        Exception: If no gpu is available

    Returns:
        list: device indexes of available gpus
    """
    state_gpus = py3nvml.get_free_gpus()
    free_gpus = [index for index, is_free in enumerate(state_gpus) if is_free]

    while wait and not free_gpus:
        time.sleep(refresh_time)
        state_gpus = py3nvml.get_free_gpus()
        free_gpus = [index for index, is_free in enumerate(state_gpus) if is_free]

    if not free_gpus:
        raise Exception("No available gpus at the moment")

    return free_gpus
