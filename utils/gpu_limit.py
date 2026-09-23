import os

import torch


def cap_gpu_memory(device):
    """Optionally cap this process's share of the GPU, for shared machines.

    Measured footprint of these trainings: ~121 MiB peak during a training
    step, ~244 MiB during evaluation, ~396 MiB held by the caching allocator,
    plus the CUDA context -- comfortably under 1 GiB in total. The cap exists
    so that if something unexpected does balloon (a batch-size change, a new
    variant), THIS process raises OOM instead of starving a co-tenant job.

    Set DLSSNET_GPU_MEM_FRACTION to a fraction of total GPU memory, e.g.
    0.05 on a 40 GB A100 = 2 GiB ceiling (~8x measured headroom). Unset =
    no cap, PyTorch's default behaviour.
    """
    frac = os.environ.get("DLSSNET_GPU_MEM_FRACTION")
    if not frac or device.type != "cuda":
        return
    frac = float(frac)
    torch.cuda.set_per_process_memory_fraction(frac, device.index or 0)
    total = torch.cuda.get_device_properties(device).total_memory / 2**20
    print(f"GPU memory capped at {frac:.3f} of {total:.0f} MiB = {frac * total:.0f} MiB for this process", flush=True)
