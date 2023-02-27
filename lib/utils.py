"""
Miscellaneous utility functions
"""
import yaml
from box import ConfigBox
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


def load_params(params_path: str) -> ConfigBox:
    """Load parameters file to be used as object

    Args:
        params_path (str): route to parameters

    Returns:
        _type_: _description_
    """
    with open(params_path, "rb") as config_file:
        params = yaml.safe_load(config_file)
        params = ConfigBox(params)
    return params


def get_available_cuda_devices(min_memory: int = 0, wait: bool = False, refresh_time: int = 10) -> list[int]:
    """get a list of available cuda devices
    with total memory over the min_memory

    Args:
        min_memory (int, optional): minimum required memory for device (in GB). Defaults to 0
        wait (bool, optional): Whether to wait until a cuda device  is free. Defaults to False.
        refresh_time (int, optional): how often to recheck if a cuda device is available (only when wait=True). Defaults to 10.

    Raises:
        IndexError: If no cuda device is available

    Returns:
        list: device indexes of available cuda devices, ordered from lowest memory to highest
    """

    print("Searching for available cuda devices")
    available_devices = []
    devices_memory = []
    nvmlInit()
    device_count = nvmlDeviceGetCount()

    for index in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(index)
        processes = nvmlDeviceGetComputeRunningProcesses(handle)
        # If process exist, the gpu is under use
        if not processes:
            # Get memory size (in B) and transform to GB
            device_memory_info = nvmlDeviceGetMemoryInfo(handle)
            device_total_memory = device_memory_info.total / 1_000_000_000
            if device_total_memory >= min_memory:
                available_devices.append(index)
                devices_memory.append(device_total_memory)
            else:
                print(f"Device {index} availabe but insuficient memory: {device_total_memory:.2f} GB")

    # Repeat process if wait and no devices have been found
    if wait and not available_devices:
        time.sleep(refresh_time)
        available_devices = get_available_cuda_devices(min_memory, wait, refresh_time)
    nvmlShutdown()

    if not available_devices:
        raise IndexError("No available cuda devices at the moment")

    # Sort devices from lowest memory to highest
    available_devices = [device for _, device in sorted(zip(devices_memory, available_devices))]

    return available_devices
