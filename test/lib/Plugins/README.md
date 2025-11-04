# Triton TTIR and TTGIR Out of Tree Plugin Passes

## Overview
Triton’s existing pass pipelines are assembled in the various extended compiler.py files that live in Triton’s backends. Currently when we want to insert
passes either for downstream optimizations, custom ops, or instrumentation it is required for the compiler.py file itself to be modified and all of Triton to be
recompiled.

In order to allow for more downstream configurability we have implemented a custom MLIR level (TTIR and TTGIR) pass plugin and configuration system that allows for either
overriding the compiler.py pipeline entirely or inserting passes and custom ops through a compiler pipeline hook. Example use cases include:
- Custom ops and lowering passes
- Custom optimization passes
- Instrumentation passes
- Specialized per kernel passes (e.g. kernel/model specific warp specialization)

Custom passes/ops are implemented as a shared library that is loaded by Triton at JIT compile/runtime. The plugins can be implement entirely out of tree or in the Triton source tree as
long as the plugin is linked into the libtriton.so and the Triton include passes are used to build the plugin.

## Example 1: Developing a custom pass and running triton-opt to inspect the modified IR
``` bash
export LLVM_BUILD_SHARED_LIBS=1;  make dev-install-llvm
TRITON_PASS_PLUGIN_PATH=libTritonPluginsTestLib.so triton-opt -tritongpu-plugin test/Plugins/test-plugin.mlir
```
``` MLIR
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:80"} {
  tt.func @foo() {
    tt.return
  }
}
```

After the out of tree pass runs, becomes:
``` MLIR
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:80"} {
  tt.func @bar() {
    tt.return
  }
}
```
Function "foo" is renamed to "bar" by the out of tree pass.

## Example 2: Inserting a new pass into the compiler pipeline
Let's take the following toy kernel example:
``` python
import torch
import os

import triton
import triton.language as tl
from triton._C.libtriton import ir, passes
from triton import knobs

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def kernel1(BLOCK_SIZE: tl.constexpr):
    return

if __name__ == '__main__':

    size = 98432
    x = torch.rand(size, device=DEVICE)
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    h = kernel1[grid](BLOCK_SIZE=1024)
    print(h.asm["ttgir"])
```

Running as is will produce the expected output of print the TTGIR of the kernel:
``` bash
python test.py
```
``` MLIR
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kernel1() attributes {noinline = false} {
    tt.return loc(#loc1)
  } loc(#loc)
} loc(#loc)
#loc = loc("/home/triton/test.py":13:0)
#loc1 = loc("/home/triton/test.py":14:4)
```

Running with same code but loading the plugin library also produces the same results since, while the plugin pass has been loaded and registered with the
pass manager is not inserted into the compiler pass pipeline:

``` bash
TRITON_ALWAYS_COMPILE=1 TRITON_PASS_PLUGIN_PATH=/home/triton/python/triton/plugins/libTritonPluginsTestLib.so python test.python
```

``` MLIR
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kernel1() attributes {noinline = false} {
    tt.return loc(#loc1)
  } loc(#loc)
} loc(#loc)
#loc = loc("/home/triton/test.py":13:0)
#loc1 = loc("/home/triton/test.py":14:4)
```

Finally, if we both load the plugin at runtime and insert the pass pipeline hook into the kernel code:

``` python
import torch
import os

import triton
import triton.language as tl
from triton._C.libtriton import ir, passes
from triton import knobs

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def kernel1(BLOCK_SIZE: tl.constexpr):
    return

def inspect_stages_hook(self, stages, options, language, capability):

    def make_ttir_wrapper(mod, metadata, opt, capability):
        mod = self.make_ttir(mod, metadata, opt, capability)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.plugin.add_plugin(pm)
        pm.run(mod, 'make_ttir_plugin')
        return mod

    stages["ttir"] = lambda src, metadata: make_ttir_wrapper(src, metadata, options, capability)

if __name__ == '__main__':

    size = 98432
    x = torch.rand(size, device=DEVICE)
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    knobs.runtime.add_stages_inspection_hook = inspect_stages_hook
    h = kernel1[grid](BLOCK_SIZE=1024)
    print(h.asm["ttgir"])
```

``` bash
TRITON_ALWAYS_COMPILE=1 TRITON_PASS_PLUGIN_PATH=/home/triton/python/triton/plugins/libTritonPluginsTestLib.so python test.py
```

Shows the pass ran and modified the kernel name.

``` MLIR
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @foo() attributes {noinline = false} {
    tt.return loc(#loc1)
  } loc(#loc)
} loc(#loc)
#loc = loc("/home/triton/test.py":13:0)
#loc1 = loc("/home/triton/test.py":14:4)
```

The hook, as it's defined will insert the pass at the vert end of the make_ttir pipeline.
This functionality can be toggled on and off by just commenting out this line in kernel code:
knobs.runtime.add_stages_inspection_hook = inspect_stages_hook
without needing any core compiler changes or rebuilding Triton.
