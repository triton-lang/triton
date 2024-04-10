## 1. Summary
This PR is a **basic** POC of using a light C++ wrapper to reduce `triton` kernel launching overhead. I hope to get some feedbacks from the community before I invest more of my own time.

## 2. Reduce launching overhead
To be fair, `triton` runtime does not incur a very bad overhead. However, when inputs are small, like inferencing with kv-cache, a kernel can finish before launching the next kernel. What's worse is the overhead is proportional to number of arguments a kernel has (see benchmarks below).

The source of this overhead is `JITFunction.run` method. It creates a lot of python containers, and has some very expensive function calls:
https://github.com/openai/triton/blob/f5722cb9d8a8ce6211c77b85c340391e3e9c78e0/python/triton/runtime/jit.py#L401-L402

https://github.com/openai/triton/blob/f5722cb9d8a8ce6211c77b85c340391e3e9c78e0/python/triton/runtime/jit.py#L416-L419

The idea is pretty simple: move this function to C++ to minimize calls to python runtime. [tmp/low_latency_jit_function.cpp](https://github.com/openai/triton/compare/main...liboyue:low-latency-jit-function?expand=1#diff-d98237084011241c6cb3c928f58ea1bacb3f63a77c2615e3f5c20a520a576b40) implements part of the `run` function in C++,
[tmp/low_latency_jit.py](https://github.com/openai/triton/compare/main...liboyue:low-latency-jit-function?expand=1#diff-cff24374412e114f8fcec694226bf02d2f9348ba80298156b6db80091fad8aa1) wraps around it. [tmp/profile.py](https://github.com/openai/triton/compare/main...liboyue:low-latency-jit-function?expand=1#diff-f98cf7cc75ef0a9d0e2fb8c31a3c72ec94760154ce7fb33d4d739c721ec0e0db) is the benchmarking code.

## 3. Benchmarks
The benchmarks measures a forward and a backward run for 6 functions:
- noop: this function does nothing so we can find the resolution of our timing code.
- baseline: the op using PyTorch (just `(x * 2).backward(dy)`)
- short_params: a `autograd` function doing the same thing but calls a forward and a backward `triton` kernels under the hood.
- long_params: the same as short_params but with a lot of useless params.
- short_params_optimized and long_params_optimized: the same `triton` kernels compiled by `low_latency_jit.jit`.

CPU: Xeon 6154 @ 3.00GHz
GPU: RTX 2080 SUPER
CUDA: 12.1
PyTorch: 2.2.2

### 3.1 PyTorch forward and backward
Run time in microseconds of 10 runs, where each function is run 5000 steps per run:
```
       noop  baseline  short_params  long_params  short_params_optimized  long_params_optimized
count 10.00     10.00         10.00        10.00                   10.00                  10.00
mean   1.41     72.63        185.60       258.91                   76.71                  83.59
std    0.01      0.07         19.89        22.92                    1.93                   6.04
min    1.39     72.58        167.03       235.47                   75.04                  75.54
25%    1.41     72.59        170.12       244.80                   75.09                  77.79
50%    1.41     72.61        176.35       247.28                   75.72                  84.85
75%    1.42     72.63        195.56       277.37                   78.37                  87.37
max    1.43     72.81        218.99       298.59                   80.18                  92.60
```
From the measurements, we can see the resolution is around 1.4us, optimized results are about as fast as the PyTorch baseline, and much much faster than `triton` shipped with PyTorch 2.2.2.

### 3.2 Overhead
This benchmark compares only the forward kernels. The time is measured by the begin and end CUDA events without clearing L2 cache.

"kernel" means running the compiled kernels directly like `fwd.cache[0][k].c_wrapper(***)`, which is the actual kernel run time plus some python overhead. "_warmup" appendix means the kernel is called by `fwd[(grid,](args, warmpup=True)`, which is the python time.

Run time in microseconds:
```
       noop  kernel  short_warmup  long_warmup  short_optimized_warmup  long_optimized_warmup  short  long  short_optimized  long_optimized
count 10.00   10.00         10.00        10.00                   10.00                  10.00  10.00 10.00            10.00           10.00
mean   0.06   20.27         30.77        46.49                    8.26                  11.11  44.96 61.26            20.28           23.27
std    0.00    0.00          0.15         0.15                    0.11                   0.12   0.18  0.15             0.00            0.12
min    0.06   20.27         30.66        46.24                    8.16                  11.04  44.72 61.14            20.27           23.14
25%    0.06   20.27         30.68        46.45                    8.19                  11.05  44.86 61.16            20.28           23.18
50%    0.06   20.27         30.74        46.48                    8.25                  11.08  44.96 61.18            20.28           23.21
75%    0.06   20.27         30.77        46.53                    8.27                  11.10  45.03 61.29            20.28           23.35
max    0.06   20.28         31.18        46.82                    8.56                  11.44  45.38 61.60            20.29           23.49
```
So the CPU overhead of optimized jit function can be completely hidden. I'm pretty sure the overhead can still be improved with a bit more refactoring.

## Next steps:
- [ ] Add tests
- [ ] Optimize current code (need to remove unnecessary calls to python runtime)
- [ ] Support kwargs
- [ ] Refactor `run` function to allow better type hints, so that we can further reduce python calls
- [ ] Refactor cache management (currently the kernel cache is a two-level dictionary, we can change it to a one-level dict in C++)
- [ ] Move these files to correct folders
- [ ] Add methods to access C++ data structures from Python
- [ ] Refactor kernel wrapper to launch kernels from C++

