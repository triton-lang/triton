# ISAAC

This is the developer repository for ISAAC, a library that uses machine learning to find input-aware kernels for element-wise operations, 1D/2D reductions and GEMM. It works with both cuBLAS and clBLAS. It's super easy to compile (no dependency!), to install (just link against libisaac.so instead of clBLAS or cuBLAS!), almost always outperforms (tuned) clBLAS and often outperforms cuBLAS. Try it!

### License

ISAAC is distributed under the GNU LGPL v2.1 License.

### Installation

ISAAC is dependency-free, and will load either OpenCL and/or CUDA 7.0+ _dynamically_ depending on which GPUs are detected at runtime.

You only need CMake 2.8.7+ and a C++11 compliant compiler:  
 

```
git clone https://github.com/ptillet/isaac.git
mkdir -p isaac/build && cd isaac/build
cmake ../ && make -j4
```

Link against libisaac.so instead of libcublas.so or libclblas.so, and you're good to go! 

The C++ and Python API does some kernel fusion, but is not entirely stable. It works well to compose element-wise operations, though.


### Benchmark

```
./bench/bench-blas OP
```
where OP is axpy, gemv or gemm. It detects clBLAS or cuBLAS and compares it against ISAAC for DeepBench, Covariance and LAPACK (packed rank1 updates).

Below is what you get for a Pascal Titan X (TFLOPS). Numbers in bold represent speed-ups greater than 5%.

| BENCH  | M    | N     | K    | AT | BT | ISAAC     | cuBLAS     |
| -------| -----| ------| -----| ---| ---| ----------| -----------|
| Deep   | 1760 | 16    | 1760 | N  | N  | 1.15      | **1.65**   |
| -      | 1760 | 32    | 1760 | N  | N  | **2.43**  | 1.88       |
| -      | 1760 | 64    | 1760 | N  | N  | **3.83**  | 2.58       |
| -      | 1760 | 128   | 1760 | N  | N  | **5.53**  | 4.83       |
| -      | 1760 | 7000  | 1760 | N  | N  | 11.59     | 11.36      |
| -      | 2048 | 16    | 2048 | N  | N  | **1.56**  | 1.40       |
| -      | 2048 | 32    | 2048 | N  | N  | **2.92**  | 2.25       |
| -      | 2048 | 64    | 2048 | N  | N  | **4.32**  | 2.4        |
| -      | 2048 | 128   | 2048 | N  | N  | 5.89      | 5.74       |
| -      | 2048 | 7000  | 2048 | N  | N  | 11.89     | 11.77      |
| -      | 2560 | 16    | 2560 | N  | N  | **1.88**  | 1.28       |
| -      | 2560 | 32    | 2560 | N  | N  | **3.55**  | 2.52       |
| -      | 2560 | 64    | 2560 | N  | N  | **5.15**  | 3.01       |
| -      | 2560 | 128   | 2560 | N  | N  | 6.62      | 6.91       |
| -      | 2560 | 7000  | 2560 | N  | N  | 11.53     | 11.35      |
| -      | 1760 | 16    | 1760 | T  | N  | **1.09**  | 0.65       |
| -      | 1760 | 32    | 1760 | T  | N  | **2.08**  | 1.46       |
| -      | 1760 | 64    | 1760 | T  | N  | **2.94**  | 2.46       |
| -      | 1760 | 128   | 1760 | T  | N  | **4.77**  | 2.87       |
| -      | 1760 | 7000  | 1760 | T  | N  | 9.39      | 9.25       |
| -      | 2048 | 16    | 2048 | T  | N  | **1.47**  | 0.93       |
| -      | 2048 | 32    | 2048 | T  | N  | **2.75**  | 2.42       |
| -      | 2048 | 64    | 2048 | T  | N  | **3.46**  | 2.62       |
| -      | 2048 | 128   | 2048 | T  | N  | **5.43**  | 4.50       |
| -      | 2048 | 7000  | 2048 | T  | N  | 11.11     | 11.02      |
| -      | 2560 | 16    | 2560 | T  | N  | **1.78**  | 0.72       |
| -      | 2560 | 32    | 2560 | T  | N  | **3.06**  | 1.72       |
| -      | 2560 | 64    | 2560 | T  | N  | **4.37**  | 2.39       |
| -      | 2560 | 128   | 2560 | T  | N  | **5.52**  | 2.86       |
| -      | 2560 | 7000  | 2560 | T  | N  | **8.67**  | 7.77       |
| -      | 1760 | 7133  | 1760 | N  | T  | 11.56     | 11.48      |
| -      | 4096 | 7133  | 4096 | N  | T  | 10.69     | 10.37      |
| Cov    | 32   | 60000 | 32   | N  | T  | **1.44**  | 0.80       |
| -      | 256  | 60000 | 256  | N  | T  | **6.43**  | 3.61       |
| Lapack | 4096 | 4096  | 32   | N  | T  | **4.91**  | 2.57       |
| -      | 3456 | 3456  | 32   | N  | T  | **4.53**  | 2.50       |
| -      | 896  | 896   | 32   | N  | T  | 1.14      | **1.37**   |

For AMD Fury:

| BENCH  | M    | N     | K    | AT | BT | ISAAC     | clBLAS    |
| -------| -----| ------| -----| ---| ---| ----------| ----------|
| Deep   | 1760 | 16    | 1760 | N  | N  | **0.62**  | 0.13      |
| -      | 1760 | 32    | 1760 | N  | N  | **0.93**  | 0.27      |
| -      | 1760 | 64    | 1760 | N  | N  | **1.10**  | 0.53      |
| -      | 1760 | 128   | 1760 | N  | N  | **2.00**  | 0.99      |
| -      | 1760 | 7000  | 1760 | N  | N  | **4.66**  | 2.71      |
| -      | 2048 | 16    | 2048 | N  | N  | **0.65**  | 0.16      |
| -      | 2048 | 32    | 2048 | N  | N  | **0.83**  | 0.31      |
| -      | 2048 | 64    | 2048 | N  | N  | **1.10**  | 0.57      |
| -      | 2048 | 128   | 2048 | N  | N  | **1.71**  | 0.87      |
| -      | 2048 | 7000  | 2048 | N  | N  | **4.39**  | 2.49      |
| -      | 2560 | 16    | 2560 | N  | N  | **0.94**  | 0.19      |
| -      | 2560 | 32    | 2560 | N  | N  | **1.24**  | 0.38      |
| -      | 2560 | 64    | 2560 | N  | N  | **1.51**  | 0.73      |
| -      | 2560 | 128   | 2560 | N  | N  | **2.30**  | 1.22      |
| -      | 2560 | 7000  | 2560 | N  | N  | **4.61**  | 2.65      |
| -      | 1760 | 16    | 1760 | T  | N  | 0.42      | **0.55**  |
| -      | 1760 | 32    | 1760 | T  | N  | 0.79      | **0.89**  |
| -      | 1760 | 64    | 1760 | T  | N  | **1.50**  | 1.14      |
| -      | 1760 | 128   | 1760 | T  | N  | **1.64**  | 1.26      |
| -      | 1760 | 7000  | 1760 | T  | N  | **4.36**  | 1.73      |
| -      | 2048 | 16    | 2048 | T  | N  | **0.37**  | 0.35      |
| -      | 2048 | 32    | 2048 | T  | N  | **0.54**  | 0.41      |
| -      | 2048 | 64    | 2048 | T  | N  | **0.93**  | 0.76      |
| -      | 2048 | 128   | 2048 | T  | N  | 0.78      | **0.96**  |
| -      | 2048 | 7000  | 2048 | T  | N  | **3.69**  | 1.29      |
| -      | 2560 | 16    | 2560 | T  | N  | 0.67      | 0.66      |
| -      | 2560 | 32    | 2560 | T  | N  | **1.10**  | 0.81      |
| -      | 2560 | 64    | 2560 | T  | N  | **1.51**  | 1.08      |
| -      | 2560 | 128   | 2560 | T  | N  | **1.84**  | 1.15      |
| -      | 2560 | 7000  | 2560 | T  | N  | **4.22**  | 1.49      |
| -      | 1760 | 7133  | 1760 | N  | T  | **4.60**  | 2.70      |
| -      | 4096 | 7133  | 4096 | N  | T  | **5.66**  | 2.62      |
| Cov    | 32   | 60000 | 32   | N  | T  | **0.34**  | 0.00      |
| -      | 256  | 60000 | 256  | N  | T  | **1.25**  | 0.19      |
| Lapack | 4096 | 4096  | 32   | N  | T  | **1.85**  | 1.23      |
| -      | 3456 | 3456  | 32   | N  | T  | **1.78**  | 1.27      |
| -      | 896  | 896   | 32   | N  | T  | 0.35      | 0.36      |

### BLAS routines supported

Currently supported functions are:

| BLAS1         | BLAS2         | BLAS3         |
| --------------| --------------| --------------|
| sAXPY         | sGEMV         | sGEMM         |
| sCOPY         | sGER          |               |
| sSCAL         |               |               |
| sDOT          |               |               |
| sASUM         |               |               |

### Contributing

You can contribute to further tuning isaac if you have one of the following architecture:
- NVidia: SM 2.x ; SM 3.5 ; SM 5.0 ; SM 6.0 
- Intel: Skylake iGPU

I'm planning on adding double precision support soon.

I'm no longer very active on this project, although I'll fix bugs. I've been working on a more ambitious project lately.
