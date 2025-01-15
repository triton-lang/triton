<div align="center">
  <img src="https://lh5.googleusercontent.com/wzQKEsTFkrgNQO9JjhGH5wFvslJr1saLtLaJ_a6Fp_gNENpvt3VG7BmztwngU9hFJaU4CPwGiw1opQtDvTkLrxWRbO_a12Q-pdESWHgtmheIHcPbOL5ZMC4TSiJVe5ty1w=w3517" alt="Triton logo">
</div>

| **`Documentation`** | **`Nightly Wheels`** |
|-------------------- | -------------------- |
| [![Documentation](https://github.com/triton-lang/triton/actions/workflows/documentation.yml/badge.svg)](https://triton-lang.org/) | [![Wheels](https://github.com/triton-lang/triton/actions/workflows/wheels.yml/badge.svg?branch=release/2.0.x)](https://github.com/triton-lang/triton/actions/workflows/wheels.yml) |

# Triton

This is the development repository of Triton, a language and compiler for writing highly efficient custom Deep-Learning primitives. The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs.

The foundations of this project are described in the following MAPL2019 publication: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). Please consider citing this work if you use Triton!

The [official documentation](https://triton-lang.org) contains installation instructions and tutorials.  See also these third-party [Triton puzzles](https://github.com/srush/Triton-Puzzles), which can all be run using the Triton interpreter -- no GPU required.

# Quick Installation

You can install the latest stable release of Triton from pip:

```shell
pip install triton
```

Binary wheels are available for CPython 3.9-3.13.

# Enabling Blackwell Support

The main branch now features support for NVIDIA Blackwell GPUs using 5th
generation tensor cores. To enable this, you will need two additional steps:

1. Build a pre-release PyTorch from source with CUDA 12.8
2. Build triton from the latest source


First, to build pytorch you need to have CUDA 12.8 installed locally. If not,
follow the [instructions for your platform](https://developer.nvidia.com/cuda-downloads)
```bash
# Clone and checkout pytorch 2.6 release candidate
git clone https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.6.0-rc9
git submodule sync
git submodule update --init --recursive -j 8

# Install build dependencies (assumes you already have a system compiler)
pip install -r requirements.txt
pip install mkl-static mkl-include wheel

# Build PyTorch (will take a long time)
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=$CUDA_HOME
export TORCH_CUDA_ARCH_LIST=Blackwell
python setup.py develop

# Optional, package build into a wheel to install on other machines.
python setup.py bdist_wheel
ls dist  # Wheel should be output in this directory
```

Note that if you use the domain libraries (`torchvision`, `torchtext`,
`torchaudio`, etc.) these will need to be built from source as well, otherwise
their custom PyTorch extensions will not work.

Finally, follow the instructions below to install triton from source.

# Install from source

```shell
git clone https://github.com/triton-lang/triton.git
cd triton

pip install ninja cmake wheel pybind11 # build-time dependencies
pip install -e python
```

Or with a virtualenv:

```shell
git clone https://github.com/triton-lang/triton.git
cd triton

python -m venv .venv --prompt triton
source .venv/bin/activate

pip install ninja cmake wheel pybind11 # build-time dependencies
pip install -e python
```

# Building with a custom LLVM

Triton uses LLVM to generate code for GPUs and CPUs.  Normally, the Triton build
downloads a prebuilt LLVM, but you can also build LLVM from source and use that.

LLVM does not have a stable API, so the Triton build will not work at an
arbitrary LLVM version.

1. Find the version of LLVM that Triton builds against.  Check
`cmake/llvm-hash.txt` to see the current version. For example, if it says:
       49af6502c6dcb4a7f7520178bd14df396f78240c

   This means that the version of Triton you have builds against
   [LLVM](https://github.com/llvm/llvm-project) 49af6502.

2. `git checkout` LLVM at this revision.  Optionally, make additional
   modifications to LLVM.

3. [Build LLVM](https://llvm.org/docs/CMake.html).  For example, you might run

       $ cd $HOME/llvm-project  # your clone of LLVM.
       $ mkdir build
       $ cd build
       $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
       $ ninja

4. Grab a snack, this will take a while.

5. Build Triton as above, but set the following environment variables.

       # Modify as appropriate to point to your LLVM build.
       $ export LLVM_BUILD_DIR=$HOME/llvm-project/build

       $ cd <triton install>
       $ LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
         LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
         LLVM_SYSPATH=$LLVM_BUILD_DIR \
         pip install -e python

# Tips for building

- Set `TRITON_BUILD_WITH_CLANG_LLD=true` as an environment variable to use clang
  and lld.  lld in particular results in faster builds.

- Set `TRITON_BUILD_WITH_CCACHE=true` to build with ccache.

- Set `TRITON_HOME=/some/path` to change the location of the `.triton`
  directory where Triton's cache is located and downloads are stored
  during the build. By default, this is the user's home directory. It
  can be changed anytime.

- Pass `--no-build-isolation` to `pip install` to make nop builds faster.
  Without this, every invocation of `pip install` uses a different symlink to
  cmake, and this forces ninja to rebuild most of the `.a` files.

- vscode intellisense has some difficulty figuring out how to build Triton's C++
  (probably because, in our build, users don't invoke cmake directly, but
  instead use setup.py).  Teach vscode how to compile Triton as follows.

    - Do a local build. Run command `pip install -e python`
    - Get the full path to the `compile_commands.json` file produced by the build:
      `find python/build -name 'compile_commands.json' | xargs readlink -f`.
      You might get a full path similar to `/Users/{username}/triton/python/build/cmake.macosx-11.1-arm64-cpython-3.12/compile_commands.json`
    - In vscode, install the
      [C/C++
      extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools),
      then open the command palette (`Shift + Command + P` on Mac, or `Shift +
      Ctrl + P` on Windows/Linux) and open `C/C++: Edit Configurations (UI)`.
    - Open "Advanced Settings" and paste the full path to
      `compile_commands.json` into the "Compile Commands" textbox.

# Running tests

There currently isn't a turnkey way to run all the Triton tests, but you can
follow the following recipe.

```shell
# One-time setup.  Note this will reinstall local Triton because torch
# overwrites it with the public version.
$ make dev-install

# To run all tests (requires a GPU)
$ make test

# Or, to run tests without a gpu
$ make test-nogpu
```

# Tips for hacking

For detailed instructions on how to debug Triton's frontend, please refer to this [tutorial](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html). The following includes additional tips for hacking on Triton's backend.

**Helpful environment variables**

- `MLIR_ENABLE_DUMP=1` dumps the IR before every MLIR pass Triton runs, for all
   kernels. Use `MLIR_ENABLE_DUMP=kernelName` to dump for a specific kernel only.
  - Triton cache can interfere with the dump. In cases where `MLIR_ENABLE_DUMP=1` does not work, try cleaning your triton cache: `rm -r ~/.triton/cache/*`
- `MLIR_DUMP_PATH` specifies where `MLIR_ENABLE_DUMP` will dump to. If unset will dump to stderr.
- `LLVM_IR_ENABLE_DUMP=1` dumps the IR before every pass run over the LLVM IR.
- `TRITON_REPRODUCER_PATH=<reproducer_path>` will generate an MLIR reproducer file
  at `<reproducer_path>` before each MLIR compiler stage. If any of the stages fail,
  `<reproducer_path>` will be a local MLIR reproducer captured right before the failing pass.
- `TRITON_INTERPRET=1` uses the Triton interpreter instead of running on the
  GPU.  You can insert Python breakpoints in your kernel code!
- `TRITON_ENABLE_LLVM_DEBUG=1` passes `-debug` to LLVM, printing a lot of
  debugging information to stdout.  If this is too noisy, run with just
  `TRITON_LLVM_DEBUG_ONLY` instead to limit the output.

  An alternative way to reduce output noisiness is running with
  `LLVM_IR_ENABLE_DUMP=1`, extract the IR before the LLVM pass of interest, and
  then run LLVM's `opt` standalone, perhaps passing `-debug-only=foo` on the
  command line.
- `TRITON_LLVM_DEBUG_ONLY=<comma-separated>` is the equivalent of LLVM's
  `-debug-only` command-line option. This limits the LLVM debug output to
  specific pass or component names (which are specified using `#define
  DEBUG_TYPE` throughout LLVM and Triton) in order to allow the debug output to
  be less noisy. `TRITON_LLVM_DEBUG_ONLY` allows for one or more comma
  separated values to be specified (eg
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions"` or
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions,regalloc"`).
- `TRITON_ENABLE_ASAN=1` invokes the LLVM address sanitizer for
  memory leak and out of bounds access detection. Currently only supported on the AMD
  backend. This must be run using the ASAN libraries documented [here](https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/using-gpu-sanitizer.html).

  When enabling the address sanitizer it is recommended to disable various memory caching strategies
  both within the ROCm stack and PyTorch. This will give the address sanitizer the best chance at finding the
  memory fault where it originates. See this [test](https://github.com/triton-lang/triton/blob/main/third_party/amd/python/test/test_address_sanitizer.py) for more details.

- `USE_IR_LOC={ttir,ttgir}` reparses the IR such that the location information
  will be the line number of the IR file with that particular extension,
  instead of line number of the python file. This can provide a direct mapping
  from the IR to llir/ptx. When used with performance tools, it can provide a
  breakdown on IR instructions.
- `TRITON_PRINT_AUTOTUNING=1` prints out the best autotuning config and total time
  spent for each kernel after autotuning is complete.
- `DISABLE_LLVM_OPT` will disable llvm optimizations for make_llir and make_ptx
  if its value is true when parsing as Bool. Otherwise, it will be parsed as a list
  of flags to disable llvm optimizations. One usage case is
  `DISABLE_LLVM_OPT="disable-lsr"`
  Loop strength reduction is known to cause up to 10% performance changes for
  certain kernels with register pressure.
- `TRITON_ALWAYS_COMPILE=1` forces to compile kernels regardless of cache hit.
- `MLIR_ENABLE_TIMING` dumps the timing information for each MLIR pass.
- `LLVM_ENABLE_TIMING` dumps the timing information for each LLVM pass.
- `TRITON_DEFAULT_FP_FUSION` overrides the default behavior of allowing fp fusion (mul+add->fma).
- `MLIR_ENABLE_DIAGNOSTICS=<comma-separated>` controls diagnostic emission in MLIR.
  Options are: `warnings`, `remarks`, `stacktraces`, `operations`.
  Use comma-separated values to customize output. For example,
  `MLIR_ENABLE_DIAGNOSTICS=remarks,operations` enables remarks and IR operations,
  while `MLIR_ENABLE_DIAGNOSTICS=warnings,stacktraces` enables warnings with
  stacktraces. By default, only errors are shown. Setting `warnings` includes
  errors and warnings; `remarks` includes errors, warnings, and remarks.
- `MLIR_ENABLE_REMARK` is deprecated. Please use `MLIR_ENABLE_DIAGNOSTICS=remarks`.
- `TRITON_KERNEL_DUMP` enables the dumping of the IR from each compilation stage and the final ptx/amdgcn.
- `TRITON_DUMP_DIR` specifies the directory to save the dumped IR and ptx/amdgcn when `TRITON_KERNEL_DUMP` is set to 1.
- `TRITON_KERNEL_OVERRIDE` enables the override of the compiled kernel with a user-specified IR/ptx/amdgcn at the beginning of each compilation stage.
- `TRITON_OVERRIDE_DIR` specifies the directory from which to load the IR/ptx/amdgcn files when `TRITON_KERNEL_OVERRIDE` is set to 1.
- `TRITON_F32_DEFAULT` sets the default input precision of `tl.dot` when using 32-bit floats, which can be either `ieee`, `tf32`, or `tf32x3`.

**Kernel Override Steps**

```bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=<dump_dir>
export TRITON_KERNEL_OVERRIDE=1
export TRITON_OVERRIDE_DIR=<override_dir>
# Step 1: Run the kernel once to dump kernel's IRs and ptx/amdgcn in $TRITON_DUMP_DIR
# Step 2: Copy $TRITON_DUMP_DIR/<kernel_hash> to $TRITON_OVERRIDE_DIR
# Step 3: Delete the stages that you do not want to override and modify the stage you do want to override
# Step 4: Run the kernel again to see the overridden result
```


# Changelog

Version 2.0 is out! New features include:

- Many, many bug fixes
- Performance improvements
- Backend rewritten to use MLIR
- Support for kernels that contain back-to-back matmuls (e.g., flash attention)

# Contributing

Community contributions are more than welcome, whether it be to fix bugs or to add new features at [github](https://github.com/triton-lang/triton/). For more detailed instructions, please visit our [contributor's guide](CONTRIBUTING.md).

# Compatibility

Supported Platforms:

- Linux

Supported Hardware:

- NVIDIA GPUs (Compute Capability 8.0+)
- AMD GPUs (ROCm 6.2+)
- Under development: CPUs

# Development Container (Dev Container)

**Dev Containers** for the Triton project are available from
the [triton-dev-containers repository](https://github.com/redhat-et/triton-dev-containers)

### Key Benefits:
- **Consistency**: All developers can work with the same development
  environment, ensuring uniform behavior across different systems.
- **Isolation**: The container prevents potential conflicts with software
  installed on your local machine.
- **Portability**: Easily share the development environment with team members,
  minimizing onboarding time and setup issues.

### How to Use the Dev Container:

For detailed instructions on how to use the dev containers please see
the [dev container user guide](https://github.com/redhat-et/triton-dev-containers/blob/main/.devcontainer/devcontainer.md)

# Warp Specialization Support


Warp specialization enhances kernel performance by utilizing an asynchronous execution model, where different parts of the kernel are handled by separate hardware units. The data communication between these units, via shared memory on the H100, operates with high efficiency. With this in mind, we’ve developed an automatic warp specialization optimization that partitions a user kernel into asynchronous tasks (which map to warp groups on NVIDIA GPU), which naturally execute concurrently, leveraging the hardware’s multitasking warp scheduler. The following sections provide a breakdown of the compiler features developed to enable warp specialization.


## Asynchronous Tasks

Warp specialization is built on top of the concept of partitioning the user’s program into asynchronous tasks (referred to as "async tasks" or “tasks” in the following sections). Each async task will be executed by a standalone warp group on the supported hardware, to achieve instruction level parallelism. While optimally and automatically partitioning asynchronous tasks remains a challenge for compilers, our approach to automatic task partitioning has proven effective for kernels similar to typical examples like GEMM and Flash Attention.

To enable warp specialization, user just needs to specify certain autotune flags, i.e., `num_consumer_groups` and `num_buffers_warp_spec`. For example, a warp-specialized GEMM implementation might look like below. You can find a complete example in 09-persistent-matmul.py.

```python
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=4,
            num_consumer_groups=2,
            num_buffers_warp_spec=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_persistent_ws_kernel(
   a_ptr, b_ptr, c_ptr, M, N, K,
   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
   pid = tl.program_id(axis=0)
   num_pid_m = tl.cdiv(M, BLOCK_M)
   num_pid_n = tl.cdiv(N, BLOCK_N)
   pid_m = pid // num_pid_m
   pid_n = pid % num_pid_n
   offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
   offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
   offs_k = tl.arange(0, BLOCK_K)
   a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
   b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
   acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
   for k in range(0, tl.cdiv(K, BLOCK_K)):
       a = tl.load(a_ptrs)
       b = tl.load(b_ptrs)
       acc += tl.dot(a, b)
       a_ptrs += BLOCK_K * stride_ak
       b_ptrs += BLOCK_K * stride_bk
   c = acc.to(tl.float16)
   c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
   tl.store(c_ptrs, c)
```

The compiler automatically determines how to utilize one producer warp group and two consumer warp groups to execute the kernel. It begins by assigning task IDs to certain anchor operations, which influence the task assignments for the remaining operations. Once the anchor tasks are annotated, the compiler assigns the non-anchor operations to tasks as follows:

- Control dependencies exclusive to an anchor operation are included in the same task as the anchor operation.
- Data dependencies exclusive to an anchor operation are included in the same task as the anchor operation, unless they are another anchor operation.
- Control or data dependencies shared between tasks are included in all those tasks.

For the GEMM example above, the compiler computes a task scheme and annotates it in the IR using MLIR attributes. To illustrate this more clearly, let's use source code annotations. After task propagation:


```python
@triton.jit
def matmul_persistent_ws_kernel(
   a_ptr, b_ptr, c_ptr, M, N, K,
   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
   pid = tl.program_id(axis=0) # async_task 0, 1
   num_pid_m = tl.cdiv(M, BLOCK_M) # async_task 0, 1
   num_pid_n = tl.cdiv(N, BLOCK_N) # async_task 0, 1
   pid_m = pid // num_pid_m # async_task 0, 1
   pid_n = pid % num_pid_n # async_task 0, 1
   offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) # async_task 0, 1
   offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) # async_task 0, 1
   offs_k = tl.arange(0, BLOCK_K) # async_task 0
   a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak) # async_task 0
   b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn) # async_task 0
   acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) # async_task 1
   for k in range(0, tl.cdiv(K, BLOCK_K)): # async_task 0, 1
       a = tl.load(a_ptrs)   # async_task 0
       b = tl.load(b_ptrs)   # async_task 0
       acc += tl.dot(a, b)   # async_task 1
       a_ptrs += BLOCK_K * stride_ak # async_task 0
       b_ptrs += BLOCK_K * stride_bk # async_task 0
   c = acc.to(tl.float16) # async_task 1
   c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :] # async_task 1
   tl.store(c_ptrs, c) # async_task 1
```


## Data Partitioning

To further improve performance, the compiler will split the same workload across two async tasks  This way, when one task is blocked on a heavy computation (e.g., the dot operation), the other group can execute other operations in parallel. The compiler determines how to divide the work between the two tasks to maximize performance. On the H100 GPU, the compiler will, by default, attempt to split the input tensor A along the M dimension so that each consumer computes half of the output tensor independently. This approach is known as cooperative partitioning. If this split is not advantageous—for instance, if it results in a smaller-than-native `wgmma` instruction—the compiler will instead attempt to split along the N dimension.

The transformed code for the above GEMM kernel with a configured tile size [128, 256, 64] will look like below (using source annotations instead of IR for illustration).


```python
@triton.jit
def matmul_persistent_ws_kernel(
   a_ptr, b_ptr, c_ptr, M, N, K,
   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
   pid = tl.program_id(axis=0) # async_task 0, 1, 2
   num_pid_m = tl.cdiv(M, BLOCK_M) # async_task 0, 1, 2
   num_pid_n = tl.cdiv(N, BLOCK_N) # async_task 0, 1, 2
   pid_m = pid // num_pid_m # async_task 0, 1, 2
   pid_n = pid % num_pid_n # async_task 0, 1, 2
   offs_m_1 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M // 2) # async_task 0, 1, 2
   offs_m_2 = pid_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M) # async_task 0, 1, 2
   offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_N) # async_task 0, 1, 2
   offs_k = tl.arange(0, BLOCK_K) # async_task 0
   a_ptrs_1 = a_ptr + (offs_m_1[:, None] * stride_am + offs_k[None, :] * stride_ak) # async_task 0
   a_ptrs_2 = a_ptr + (offs_m_2[:, None] * stride_am + offs_k[None, :] * stride_ak) # async_task 0
   b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn) # async_task 0
   acc_1 = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32) # async_task 1
   acc_1 = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32) # async_task 2
   for k in range(0, tl.cdiv(K, BLOCK_K)): # async_task 0, 1, 2
       a_1 = tl.load(a_ptrs_1)   # async_task 0
       a_2 = tl.load(a_ptrs_2)   # async_task 0
       b = tl.load(b_ptrs)   # async_task 0
       acc_1 += tl.dot(a_1, b)   # async_task 1
       acc_2 += tl.dot(a_2, b)   # async_task 2
       a_ptrs_1 += BLOCK_K * stride_ak # async_task 0
       a_ptrs_2 += BLOCK_K * stride_ak # async_task 0
       b_ptrs += BLOCK_K * stride_bk # async_task 0
   c_1 = acc_1.to(tl.float16) # async_task 1
   c_2 = acc_2.to(tl.float16) # async_task 2
   c_ptrs_1 = c_ptr_1 + stride_cm * offs_m_1[:, None] + stride_cn * offs_n[None, :] # async_task 1
   c_ptrs_2 = c_ptr_2 + stride_cm * offs_m_2[:, None] + stride_cn * offs_n[None, :] # async_task 2
   tl.store(c_ptrs_1, c_1) # async_task 1
   tl.store(c_ptrs_2, c_2) # async_task 2
```


## Code Partitioning

We assume all operations are already marked with a list of taskIds. We first find all communications required between warp groups. Each communication starts from a load operation with a single taskId, and ends at a direct user of the load which belongs to a different taskId. For `ForOps` containing a communication channel, we add additional arguments: `phase` and `bufferIndex`.

We introduce a tuning configuration: `num_buffers_warp_spec`. For each communication channel, if it is within a `forOp`, we use an array of buffers in SMEM to save the results, and size of the array is determined by `num_buffers_warp_spec`. We also use an array of barriers for each communication channel that is inside a `ForOp`. At this pass, four new operations are introduced to correctly synchronize between the producer and the consumer: `ProducerAcquireOp`, `ProducerCommitOp`, `ConsumerWaitOp`, and `ConsumerReleaseOp`. Each of the four new ops take a token, a buffer Index. `ProducerAcquire` and `ConsumerWait` take an additional phase operand.


For `ForOps` with multiple task Ids, we clone one copy for each taskId, each copy contains the operations with the specific taskId. In the end, we create multiple `IfOps`, one for each possible taskId. We go through the body of the function, clone the op for each attached task Id and put the cloned op in the right `IfOp`.

To adjust register usage, we introduce two new ops: `RegAllocOp` and `RegDeallocOp`, both taking an integer operand. For each warp group, we decide to insert either `RegAllocOp` or `RegDeallocOp`. The current heuristic is simple: if the task Id is 0, we add `RegDeallocOp`, otherwise we use `RegAllocOp`. The amount of register adjustment can be tuned via `reg_dec_producer` and `reg_inc_consumer`.

This pass also lowers `loadOp`s to `AsyncTMACopyGlobalToLocalOp` or `AsyncCopyGlobalToLocalOp`, so the communication can be expressed via SMEM. For TMA, the producer will become
`ProducerAcquire` -> `barrier_expect` -> `AsyncTMACopyGlobalToLocalOp`, and the consumer will contain `wait_barrier` -> ops -> `ConsumerRelease`. For non-TMA loads, the producer will become `ProducerAcquire` -> `AsyncCopyGlobalToLocalOp` -> `ProducerCommitOp`, and the consumer will contain `ConsumerWaitOp` -> ops -> `ConsumerRelease`.
