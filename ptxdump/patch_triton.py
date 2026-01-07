import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
import torch
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("patch_triton")

import triton.testing

import triton.compiler.compiler

# Output directory
# PTX_DIR will be determined dynamically

logger.info(f"Patching Triton to dump PTX")

# Mock Target
def get_blackwell_target():
    return GPUTarget("cuda", 100, 32)

# Patch triton.compile
original_compile = triton.compile
def my_compile(src, target=None, options=None):
    # Force target
    target = get_blackwell_target()
    
    # Call original compile
    try:
        ret = original_compile(src, target=target, options=options)
    except Exception as e:
        logger.error(f"Compilation failed for {getattr(src, 'name', 'unknown')}: {e}")
        # We re-raise because some tests might expect compilation failure
        raise e

    # Save PTX
    if hasattr(ret, 'asm') and "ptx" in ret.asm:
        # Determine output directory from env var or default
        ptx_dir = os.environ.get("TRITON_PTX_DUMP_DIR", "/root/wkspace/triton/ptx_dump_all")
        os.makedirs(ptx_dir, exist_ok=True)

        name = getattr(src, 'name', 'kernel')
        # Sanitize name
        name = name.replace("<", "_").replace(">", "_").replace(" ", "_").replace(":", "_")
        
        # Add counter to ensure uniqueness
        if not hasattr(my_compile, "counter"):
            my_compile.counter = 0
        my_compile.counter += 1
        
        filename = os.path.join(ptx_dir, f"{name}_{my_compile.counter}.ptx")
        try:
            with open(filename, "w") as f:
                f.write(ret.asm["ptx"])
            logger.info(f"[PTX DUMP] Saved {filename}")
        except Exception as e:
            logger.error(f"Failed to save PTX for {name}: {e}")
    
    return ret

triton.compile = my_compile
triton.compiler.compiler.compile = my_compile
triton.compiler.compile = my_compile

# Patch Driver
import triton.runtime.driver
def patch_driver():
    if hasattr(triton.runtime.driver, 'active') and triton.runtime.driver.active:
        triton.runtime.driver.active.get_current_target = get_blackwell_target
        triton.runtime.driver.active.get_current_device = lambda: 0
        triton.runtime.driver.active.get_active_torch_device = lambda x=None: torch.device("cuda:0")
        # Patch utils.load_binary to avoid loading the binary on the device
        if hasattr(triton.runtime.driver.active, 'utils'):
             # Return dummy values: module, function, n_regs, n_spills, n_max_threads
             triton.runtime.driver.active.utils.load_binary = lambda name, kernel, shared, device: (None, None, 0, 0, 1024)

patch_driver()

# Patch torch.cuda
torch.cuda.get_device_capability = lambda device=None: (10, 0)
torch.cuda.is_available = lambda: True
torch.cuda.device_count = lambda: 1
torch.cuda.current_device = lambda: 0
torch.cuda.set_device = lambda device: None

# Patch CompiledKernel to avoid execution
from triton.compiler.compiler import CompiledKernel
def my_run(self, *args, **kwargs):
    # Do nothing
    return

CompiledKernel.run = my_run

# Patch Autotuner to only run the first config to speed up compilation
if hasattr(triton.runtime.autotuner.Autotuner, 'prune_configs'):
    original_prune = triton.runtime.autotuner.Autotuner.prune_configs
    def patched_prune(self, kwargs):
        configs = original_prune(self, kwargs)
        if configs:
            # print(f"Patch: Pruning configs from {len(configs)} to 1")
            return [configs[0]]
        return configs
    triton.runtime.autotuner.Autotuner.prune_configs = patched_prune

# Patch do_bench to avoid running benchmarks multiple times
def patched_do_bench(fn, warmup=None, rep=None, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean"):
    # print("Patch: Running do_bench (once)")
    fn()
    return 0.0
triton.testing.do_bench = patched_do_bench

# Also patch __getitem__ to return a runner that does nothing
# original_getitem = CompiledKernel.__getitem__
# CompiledKernel.__getitem__ = lambda self, grid: lambda *args, **kwargs: None

# We also need to patch JITFunction.run to ensure it doesn't fail when calling the kernel
# But JITFunction.run calls self.run which calls kernel.run.
# We patched CompiledKernel.run, so it should be fine.

logger.info("Triton patched for Blackwell compilation and PTX dump.")
