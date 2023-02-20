"""
Miscellaneous utility functions
"""
import subprocess


def flatten_list(input_list: list) -> list:
    """
    Flattens a nested list that contains lists as its elements.
    Only goes one level deep (i.e. works on lists of lists but not lists of lists of lists).
    """
    return [item for sublist in input_list for item in sublist]


def run_cmd(cmd: str) -> str:
    """Execute the corresponding cmd
    and get the output
    """
    cmd_out = (subprocess.check_output(cmd, shell=True)).decode("utf-8")[:-1]
    return cmd_out


def get_available_gpu_devices() -> list:
    """get available gpus

    Raises:
        Exception: if no gpu is available

    Returns:
        list: indexes of gpus
    """
    # Get a list of existing gpus (their bus name)
    info_existing_gpu = run_cmd("nvidia-smi -q -d Memory | grep -A4 GPU")
    info_existing_gpu = (info_existing_gpu.split("\n"))[1:]
    gpu_bus_ids = [gpu_bus_id.split(" ")[1] for gpu_bus_id in info_existing_gpu if "GPU" in gpu_bus_id]
    total_gpu_num = len(gpu_bus_ids)

    # Check state of the gpu buses
    info_gpus_in_use = run_cmd("nvidia-smi --query-compute-apps=gpu_bus_id --format=csv")
    gpu_bus_ids_in_use = (info_gpus_in_use.split("\n"))[1:]

    # Get the gpu index according to their bus name
    gpu_ids_in_use = []
    for bus_id in gpu_bus_ids_in_use:
        gpu_ids_in_use.append(gpu_bus_ids.index(bus_id))

    # Available gpus = gpu index not in used gpus
    available_gpus = [i for i in range(total_gpu_num) if i not in gpu_ids_in_use]

    if not available_gpus:
        raise Exception("No available cuda device at the moment")

    return available_gpus
