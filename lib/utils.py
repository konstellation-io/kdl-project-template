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


def get_available_cuda_devices(min_memory: int = 0, wait: bool = False, refresh_time: int = 10) -> list[int]:
    """get a list of available cuda devices
    with total memory over the min_memory

    Args:
        min_memory (int, optional): minimum memory necessary for our process (in GB).
                                    If cuda devices are available but do not reach the minimum requirements,
                                    a message will be printed for our information
                                    but the cuda device will NOT appear on the output. Defaults to 0
        wait (bool, optional): whether to wait for a cuda device to be available.
                               If your script MUST be runned on a cuda you may want to set this input to true.
                               We must consider that the function will run indefenitely until a device is available.
                               Therefore our script will be on lock until a cuda device is free. Defaults to False.
        refresh_time (int, optional): How often to check if a cuda device is available (in seconds).
                                      This is done to avoid overloading with queries the nvidia-smi.
                                      This value is only used if wait=True. Defaults to 10.
    Raises:
        IndexError: If no cuda device is available

    Returns:
        list: device indexes of available cuda devices, ordered from lowest memory to highest
    Usage:
        Case 1: Simple usage
        ```python
        import torch
        from lib.utils import get_available_cuda_devices
        devices = get_available_cuda_devices()
        a = torch.Tensor(5).to(device=devices[0])
        ```
        Case 2: No device found control
        If we do not chose to wait for a device to be available and none is found,
        an IndexError exception will be raise.
        You may decide to catch this error to make use of the cpu.
        ```python
        import torch
        from lib.utils import get_available_cuda_devices
        try:
            devices = get_available_cuda_devices(min_memory=15)
            device = devices[0]
        except IndexError:
            device = 'cpu'
        a = torch.Tensor(5).to(device=device)
        ```
        Case 3:
        Always wait until a dvice is available.
        ```python
        import torch
        from lib.utils import get_available_cuda_devices
        devices = get_available_cuda_devices(min_memory=15, wait=True)
        device = devices[0]
        a = torch.Tensor(5).to(device=device)
        ```
    IMPORTANT TO NOTE:
        - This function does NOT lock the cuda device.
          Meaning, that the function should be called right before the first Tensor/model is sent to device.
        - The same device can then be used for the entire process,
          therefore the function only needs to be called once per execution.
        - The longer the time between finding the cude device and sending the first Tensor,
          the more likely someone may use the selected cuda device.
        - This function does NOT work with MIG (GPU partitions)
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
