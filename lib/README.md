# Project Libraries

Importable library functions (including reusable code for ML, visualization, ...)

## utils.get_available_cuda_devices

This function can be used to get available cuda devices.

The function has three inputs:
- min_memory (int): minimum memory necessary for our process (in GB).
  If cuda devices are available but do not reach the minimum requirements,
  a message will be printed for our information
  but the cuda device will NOT appear on the output.
- wait (bool): whether to wait for a cuda device to be available.
  If your script MUST be runned on a cuda you may want to set this input to true.
  We must consider that the function will run indefenitely until a device is available.
  Therefore our script will be on lock until a cuda device is free.
- refresh_time (int): How often to check if a cuda device is available (in seconds).
  This is done to avoid overloading with queries the nvidia-smi.

The output is a list of indexes for the cuda devices, this can be used directly with torch.

```python
import torch

from lib.utils import get_available_cuda_devices

devices = get_available_cuda_devices(min_memory=15)
a = torch.Tensor(5).to(device=devices[0])
```

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

IMPORTANT TO NOTE:
- This function does NOT lock the cuda device.
  Meaning, that the function should be called right before the first Tensor/model is sent to device.
- The same device can then be used for the entire process,
  therefore the function only needs to be called once per execution.
- The longer the time between finding the cude device and sending the first Tensor,
  the more likely someone may use the selected cuda device.
- This function does NOT work with MIG (GPU partitions)
