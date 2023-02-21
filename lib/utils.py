"""
Miscellaneous utility functions
"""
import time

from pynvml import (
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
    nvmlShutdown,
)


def flatten_list(input_list: list) -> list:
    """
    Flattens a nested list that contains lists as its elements.
    Only goes one level deep (i.e. works on lists of lists but not lists of lists of lists).
    """
    return [item for sublist in input_list for item in sublist]


def get_available_cuda_devices(wait: bool = False, refresh_time: int = 10, min_memory: int = -1) -> list[int]:
    """get a list of available gpus

    Args:
        wait (bool, optional): Wait until a gpu is free. Defaults to False.
        refresh_time (int, optional): how often to recheck if a gpu is available (only when wait=True). Defaults to 10.
        min_memory (int, optional): minimum required memory for device (in GB). Defaults to -1

    Raises:
        Exception: If no gpu is available

    Returns:
        list: device indexes of available gpus
    """
    print("Searching for available cuda devices")
    available_devices = []
    nvmlInit()
    device_count = nvmlDeviceGetCount()

    for index in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(index)
        processes = nvmlDeviceGetComputeRunningProcesses(handle)
        # If existing process, the gpu is under use
        if not processes:
            # Get memory size (in B) and transform to GB
            device_memory_info = nvmlDeviceGetMemoryInfo(handle)
            device_total_memory = device_memory_info.total / 1_000_000_000
            if device_total_memory >= min_memory:
                available_devices.append(index)
            else:
                print(f"Device {index} availabe but insuficient memory: {device_total_memory} GB")

    if wait and not available_devices:
        time.sleep(refresh_time)
        available_devices = get_available_cuda_devices(wait, refresh_time, min_memory)
    nvmlShutdown()

    if not available_devices:
        raise Exception("No available cuda devices at the moment")

    return available_devices
