"""Bind real GPU memory before loading native SE; preserve the chosen entry CLI."""
import json
import os
from pathlib import Path
import runpy
import sys

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable inside SE boundary")
properties = torch.cuda.get_device_properties(0)
limit = 9 * 1024 ** 3
if properties.total_memory < limit:
    raise RuntimeError("selected GPU cannot provide the declared 9 GiB candidate cap")
torch.cuda.set_per_process_memory_fraction(limit / properties.total_memory, 0)
torch.set_num_threads(8)
receipt_path = Path(os.environ["SE_E2_OS_RECEIPT"])
receipt = json.loads(receipt_path.read_text())
receipt.update({"gpu_name": properties.name, "gpu_total_bytes": properties.total_memory,
                "torch_allocator_limit_bytes": limit, "torch_threads": torch.get_num_threads()})
receipt_path.write_text(json.dumps(receipt, indent=2))
target = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(target).parent))
runpy.run_path(target, run_name="__main__")
