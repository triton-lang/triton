# Triton TTIR and TTGIR Out of Tree Plugin Passes

## Overview
Triton’s existing pass pipelines are assembled in the various extended compiler.py files that live in Triton’s backends. Currently when we want to insert
passes either for downstream optimizations, custom ops, or instrumentation it is required for the compiler.py file itself to be modified and all of Triton to be
recompiled.

In order to allow for more downstream configurability we have implemented a custom MLIR level (TTIR and TTGIR) pass plugin and configuration system that allows for either
overriding the compiler.py pipeline entirely or inserting passes and custom ops through a compiler pipeline hook. Example use cases include:
- Custom ops and lowering passes
- Custom optimization passes
- Instrumentation and analysis passes
- Specialized per kernel passes (e.g. kernel/model specific warp specialization)

Custom passes/ops are implemented as a shared library that is loaded by Triton at JIT compile/runtime. The plugins can be implement entirely out of tree or in the Triton source tree as
long as the libtriton.so is linked to the plugin and the Triton include passes are used to build the plugin.

## Example 1: Developing a custom pass and running triton-opt to inspect the modified IR
``` bash
export LLVM_BUILD_SHARED_LIBS=1;  make dev-install-llvm
TRITON_PASS_PLUGIN_PATH=/home/triton/python/triton/plugins/libTritonPluginsTestLib.so triton-opt -tritongpu-plugin test/Plugins/test-plugin.mlir
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
def kernel(BLOCK_SIZE: tl.constexpr):
    return

if __name__ == '__main__':

    size = 98432
    x = torch.rand(size, device=DEVICE)
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    h = kernel[grid](BLOCK_SIZE=1024)
    print(h.asm["ttgir"])
```

Running as is will produce the expected output of printing the TTGIR of the kernel:
``` bash
python test.py
```
``` MLIR
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kernel() attributes {noinline = false} {
    tt.return loc(#loc1)
  } loc(#loc)
} loc(#loc)
#loc = loc("/home/triton/test.py":13:0)
#loc1 = loc("/home/triton/test.py":14:4)
```

Running same code but loading the plugin library also produces the same results since, while the plugin pass has been loaded and registered with the
pass manager it is not inserted into the compiler pass pipeline:

``` bash
TRITON_PASS_PLUGIN_PATH=/home/triton/python/triton/plugins/libTritonPluginsTestLib.so python test.py
```

``` MLIR
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kernel() attributes {noinline = false} {
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
def kernel(BLOCK_SIZE: tl.constexpr):
    return

#These two methods must be implemented by the plugin
def get_key():
    return pathlib.Path(__file__).read_text()
def get_hash():
    return hashlib.sha256(get_key().encode('utf-8')).hexdigest()

def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
    # If the hook is called with no arguments we assume were just after the key and hash and don't want to
    # actually execute the pipeline yet.
    # This no argument early return must be implemented.
    if all(arg is None for arg in (stages, options, language, capability)):
        return get_key(), get_hash()

    def make_ttir_wrapper(mod, metadata, opt, capability):
        mod = self.make_ttir(mod, metadata, opt, capability)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.plugin.add_plugin(pm)
        pm.run(mod, 'make_ttir_plugin')
        return mod

    stages["ttir"] = lambda src, metadata: make_ttir_wrapper(src, metadata, options, capability)

    return get_key(), get_hash()

if __name__ == '__main__':

    size = 98432
    x = torch.rand(size, device=DEVICE)
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    h = kernel[grid](BLOCK_SIZE=1024)
    print(h.asm["ttgir"])

    if "TRITON_PASS_PLUGIN_PATH" in os.environ:
      knobs.runtime.add_stages_inspection_hook = inspect_stages_hook
    h = kernel[grid](BLOCK_SIZE=1024)
    print(h.asm["ttgir"])

    # Unset the hook to go back to the standard pipeline
    knobs.runtime.add_stages_inspection_hook = None
    h = kernel[grid](BLOCK_SIZE=1024)
    print(h.asm["ttgir"])
```

``` bash
TRITON_PASS_PLUGIN_PATH=/home/triton/python/triton/plugins/libTritonPluginsTestLib.so python test.py
```

Shows the pass ran and modified the kernel name but only after the hook is set. Any kernels before the hook or after the hook is unset are left unchanged.

``` MLIR
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kernel() attributes {noinline = false} {
    tt.return loc(#loc1)
  } loc(#loc)
} loc(#loc)
#loc = loc("/home/triton/test.py":13:0)
#loc1 = loc("/home/triton/test.py":14:4)

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @foo() attributes {noinline = false} {
    tt.return loc(#loc1)
  } loc(#loc)
} loc(#loc)
#loc = loc("/home/triton/test.py":13:0)
#loc1 = loc("/home/triton/test.py":14:4)

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kernel() attributes {noinline = false} {
    tt.return loc(#loc1)
  } loc(#loc)
} loc(#loc)
#loc = loc("/home/triton/test.py":13:0)
#loc1 = loc("/home/triton/test.py":14:4)
```

The hook, as defined, in the example will insert the pass at the end of the make_ttir pipeline but it's placement in the Triton pipeline is abritary.
This functionality can be toggled on and off by just commenting out this line in kernel code (or setting to None):
knobs.runtime.add_stages_inspection_hook = inspect_stages_hook
without needing any core compiler changes or rebuilding Triton.

## Example 3: Inserting a new pass into the compiler pipeline at an arbitary point.

Example 2 added a new pass to the end of the ttgir "stage". However the plugin pass's location is arbitary and can be dynamically inserted anywhere in the pipeline. Replacing the inspect_stages_hook function from example 2 instead with:

```python
def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
    if all(arg is None for arg in (stages, options, language, capability)):
        return get_key(), get_hash()
    module_name = 'dynamic_module'
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    stage_src = textwrap.dedent(inspect.getsource(self.make_ttir))
    stage_src = 'from triton._C.libtriton import ir, passes, llvm, amd, nvidia\n' + stage_src
    # Inject plugin pass right after loop unroll in the dynamically loaded stage source
    stage_src = stage_src.replace(
        "passes.ttir.add_loop_unroll(pm)",
        "passes.ttir.add_loop_unroll(pm)\n    passes.plugin.add_plugin(pm)"
    )
    exec(stage_src, module.__dict__)
    make_lambda = lambda f: lambda src, metadata: f(src, metadata, options, capability)
    stages["ttir"] = make_lambda(module.make_ttir)
    return get_key(), get_hash()
```
directs the new pass's placement based on other surrounding passes. Knowing which passes are in the pipeline a priori can challenging, therefore in the next example we show how to dump and inspect the entire pipeline that is run for a particlar kernel to allow for precise placement of specialized out of tree passes even if the upstream pass pipeline structure changes.

## Example 4: Fully customizing the compiler pipeline with pass and op insertions at abitrary locations

Here we now run two kernels one with the full standard Triton pipeline and one with fully customized pipeline entirely from within
kernel code with modifying any core Triton compiler code or recompiling. We run the kernel with a hook to output the standard pipeline, modify
the compiler.py file to insert our out of tree pass before add_loop_unroll pass (although there is no restriction of where it can be inserted),
then run the second kernel with a different pipeline. This modification can, as before, be seen in the kernel function name modification by the
inserted pass.

``` python
import torch
import os
import sys

import triton
import triton.language as tl
from triton._C.libtriton import ir, passes
from triton import knobs
import inspect
from importlib.util import module_from_spec, spec_from_file_location

from triton.backends.compiler import Language

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def kernel1(BLOCK_SIZE: tl.constexpr):
    return
@triton.jit
def kernel2(BLOCK_SIZE: tl.constexpr):
    return

def get_key():
    return pathlib.Path(__file__).read_text()
def get_hash():
    return hashlib.sha256(get_key().encode('utf-8')).hexdigest()

def dump_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
  if all(arg is None for arg in (stages, options, language, capability)):
      return get_key(), get_hash()
    source_code = "# This is generated from Triton compiler.py"
    source_code = (
        source_code
        + "\n"
        + "from triton._C.libtriton import ir, passes, llvm, amd, nvidia"
    )
    source_code = source_code + "\n" + "class GPUOverrideBackend:"
    source_code = source_code + "\n" + inspect.getsource(self.make_ttir)
    source_code = source_code + "\n" + inspect.getsource(self.make_ttgir)

    with open("compiler_override.py", "w") as file:
        file.write(source_code)
  return get_key(), get_hash()
def override_stages(self=None, stages=None, options=None, language=None, capability=None):
  if all(arg is None for arg in (stages, options, language, capability)):
      return get_key(), get_hash()
    if language != Language.TRITON:
        return
    full_name = "compiler_override.py"

    print(f"\nOverriding compile pass stages with file {full_name}")
    module_name = "triton_override_compiler_stages"
    spec = (
        spec_from_file_location(module_name, full_name)
        if os.path.isfile(full_name)
        else None
    )
    if not spec:
        return

    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "GPUOverrideBackend"):
        return
    module = getattr(module, "GPUOverrideBackend")

    has_func = lambda mod, name: hasattr(mod, name) and callable(getattr(mod, name))
    make_lambda = lambda f: lambda src, metadata: f(src, metadata, options, capability)
    if has_func(module, "make_ttir"):
        stages["ttir"] = make_lambda(module.make_ttir)
    if has_func(module, "make_ttgir"):
        stages["ttgir"] = make_lambda(module.make_ttgir)
    return get_key(), get_hash()

if __name__ == '__main__':

    size = 98432
    x = torch.rand(size, device=DEVICE)
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    knobs.runtime.add_stages_inspection_hook = dump_stages_hook
    h = kernel1[grid](BLOCK_SIZE=1024)
    filename = "compiler_override.py"

    with open(filename, "r") as infile:
        file_str = infile.readlines()

    with open(filename, "w") as outfile:
        for line in file_str:
            if "add_loop_unroll" in line:
                outfile.write("\n        passes.plugin.add_plugin(pm)\n")
            outfile.write(line)
    if "TRITON_PASS_PLUGIN_PATH" in os.environ:
      knobs.runtime.add_stages_inspection_hook = override_stages
    h = kernel2[grid](BLOCK_SIZE=1024)
    print(h.asm["ttgir"])
```
