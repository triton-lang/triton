## 1. Summary
This PR is a basic POC of how to reduce triton kernel launching overhead. Previous description can be found in https://github.com/liboyue/triton/tree/low-latency-jit-function/tmp/README.md

Previously I though Python is the bottleneck. But new experiments show that a pure Python frontend can still be very efficient. I noticed `triton` `main` branch has a launching overhead regression.

## 2. Sources of launching overhead
To be fair, triton runtime does not incur a very bad overhead. However, when inputs are small, like inferencing with kv-cache, a kernel can finish before launching the next kernel. What's worse is the overhead is proportional to number of arguments a kernel has (see benchmarks below).

### 2.1. `JITFunction.run`

One source of overhead is `JITFunction.run` method. It creates a lot of python containers, and has some very expensive function calls:
https://github.com/openai/triton/blob/f5722cb9d8a8ce6211c77b85c340391e3e9c78e0/python/triton/runtime/jit.py#L401-L402

https://github.com/openai/triton/blob/f5722cb9d8a8ce6211c77b85c340391e3e9c78e0/python/triton/runtime/jit.py#L416-L419

What's worse is the overhead is proportional to number of parameters of a kernel.

### 2.2. `CompiledKernel.runner`
I suspect the other major source of overhead is from calling `CompiledKernel`.

https://github.com/openai/triton/blob/f5722cb9d8a8ce6211c77b85c340391e3e9c78e0/python/triton/compiler/compiler.py#L352-L359



## 3. Benchmarks
The benchmarks measures run time for 3 `JITFunction.run` implementations:
 - default: `triton`'s implementation.
 - python: an optimized python implementation.
 - cpp: an optimized C++ implementation (not updated because Python is good enough).

Two additional functions are also measured:
 - noop: an empty op, to measure the resolution of benchmarks
 - kernel: the actual CUDA kernel with c wrapper. This is as close as possilbe to run bare kernels.

Two running modes are measured:
 - warmup: calling kernels with `kernel[grid](..., warmup=True)`, which is the Python overhead in `JITFunction.run()` function.
 - (empty): calling kernels normally.

Environment:
 - CPU: Xeon 6154 @ 3.00GHz
 - GPU: RTX 2080 SUPER
 - CUDA: 12.1
 - PyTorch: 2.2.2

Figures are run time for different input lengths.

**NOTE: the comparison is not fair, because I set the `device_type` to `"cuda"` manually, so my code is definitely much much faster.**
### 3.1. Triton 2.0

Kernel launching overhead in us (data from 33 runs)

|       |   default_short_warmup |   default_long_warmup |   python_short_warmup |   python_long_warmup |   cpp_short_warmup |   cpp_long_warmup |
|:------|-----------------------:|----------------------:|----------------------:|---------------------:|-------------------:|------------------:|
| mean  |              38.5711   |             61.3383   |               7.01908 |           10.0304    |          7.45927   |        10.2434    |
| std   |               0.117177 |              0.636091 |               0.17726 |            0.0643082 |          0.0376316 |         0.0715353 |

Kernel run time vs. input length
![kernel_time_triton_2](https://github.com/openai/triton/assets/5857249/cc4b57aa-de0a-4bd4-9ee4-626beec3f93c)
For some reasons the Python implementation is faster than the C++ implementation. But I didn't have time to figure out -- Python is fast enough.

Now the total overhead is around 15us. `JITFunction.run()` costs around 7us, then `CompiledKernel.runner` probably costs another 7us.

### 3.2. Triton 3.0

Kernel launching overhead in us (data from 33 runs)

|       |   default_short_warmup |   default_long_warmup |   python_short_warmup |   python_long_warmup |
|:------|-----------------------:|----------------------:|----------------------:|---------------------:|
| mean  |             101.073    |             158.28    |             98.1783   |            154.857   |
| std   |               0.596246 |               1.00237 |              0.480793 |              1.11324 |

Kernel run time vs. input length
![kernel_time_triton_3](https://github.com/openai/triton/assets/5857249/73c53ca2-8aeb-431d-93f1-d7b260dc2a64)
We can observe the launching overhead regression.

## 4. Proposed solutions
I believe the kernel launching overhead can be further reduced with the following simple optimizations.

### 4.1. Stronger assumptions on devices

It is very expensive to figure out which the `device_type` should be.

For example, I guess it is ok for `triton` to assume no one will have NVIDIA and AMD GPUs on the same machine. Then, the `device_type` can be cached at `triton`'s initialization time: if there is an NVIDIA GPU then `"cuda"` else ... (idk which types are supported). Although this is not a future-proof solution, I believe it is reasonable to make some strong assumptions for now.

### 4.2. Dynamically generate `run()` and `runner()`
It is very expensive to call `signature.bind()` and pack call args. Generating these functions at `jit` time can eliminate these expensive calls which can save a good amount of time.

For example, define a kernel as
```
@triton.jit
def kernel(
    a,
    b: float,
    c: tl.int,
    d: tl.tensor,
    e: tl.tensor[tl.float32],
    NUM_BLOCKS: tl.constexpr[tl.int] = 10
):
    ...
```

The generated `run()` function's signature can be (type hints are useless here so omitted)
```
def run(
    self,
    a,
    b,
    c,
    d,
    e,
    NUM_BLOCKS=10,
    *,
    grid=None,
    num_warps=None,
    num_ctas=1,
    num_stages=None,
    enable_warp_specialization=False,
    enable_fp_fusion=True,
    extern_libs=None,
    stream=None,
    warmup=False,
    device=None,
    device_type=None
):

    assert type(b) == float
    assert torch.is_tensor(d)
    assert torch.is_tensor(e)
    assert e.dtype == torch.float32
```

In this way, Python parses params and sets default values, which is much faster than `signature.bind()`.

Furthermore, `sig_key`, `constexpr_key`, `spec_key`, etc, can all be written explicitly as tuples of `run()`'s arguments. `c_wrapper`'s args can also be "hard-coded" in the same way.

### 4.3. Improving type hints

With improved type hints, kernel definitions are more informative, so that the generated `run()` functions can rely less on Python runtime and even perform type checks. This reduces overhead and provides some more safety (maybe?).

(see previous subsection for the example)

