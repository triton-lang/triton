# ISAAC

This is the developer repository for ISAAC, a library that uses machine learning to find input-aware kernels for element-wise operations, 1D/2D reductions and GEMM. It works with both cuBLAS and clBLAS. It's super easy to compile (no dependency!), to install (just link against libisaac.so instead of clBLAS or cuBLAS!), almost always outperforms (tuned) clBLAS and often outperforms cuBLAS. Try it!

### License

ISAAC is distributed under the GNU LGPL v2.1 License.

### Installation

ISAAC is dependency-free, and will load either OpenCL or CUDA 7.0+ _dynamically_ depending on which GPUs are detected at runtime.

Installation requires CMake 2.8.7+ and a C++11 compliant compiler:  
 

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

Here's what you get for  on a Pascal Titan X (TFLOPS):

| BENCH  | M    | N     | K    | AT | BT | ISAAC     | cuBLAS     |
| -------| -----| ------| -----| ---| ---| ----------| -----------|
| Deep   | 1760 | 16    | 1760 | N  | N  | 1.31      | **1.65**   |
| Deep   | 1760 | 32    | 1760 | N  | N  | **2.29**  | 1.71       |
| Deep   | 1760 | 64    | 1760 | N  | N  | **3.35**  | 2.34       |
| Deep   | 1760 | 128   | 1760 | N  | N  | **5.04**  | 4.46       |
| Deep   | 1760 | 7000  | 1760 | N  | N  | 8.16      | **10.10**  |
| Deep   | 2048 | 16    | 2048 | N  | N  | **1.60**  | 1.30       |
| Deep   | 2048 | 32    | 2048 | N  | N  | **2.80**  | 1.93       |
| Deep   | 2048 | 64    | 2048 | N  | N  | **3.69**  | 2.20       |
| Deep   | 2048 | 128   | 2048 | N  | N  | 4.84      | **5.07**   |
| Deep   | 2048 | 7000  | 2048 | N  | N  | 8.30      | **10.39**  |
| Deep   | 2560 | 16    | 2560 | N  | N  | **1.87**  | 1.27       |
| Deep   | 2560 | 32    | 2560 | N  | N  | **3.45**  | 2.25       |
| Deep   | 2560 | 64    | 2560 | N  | N  | **4.82**  | 2.71       |
| Deep   | 2560 | 128   | 2560 | N  | N  | 5.08      | 6.25       |
| Deep   | 2560 | 7000  | 2560 | N  | N  | 8.32      | 10.11      |
| Deep   | 1760 | 16    | 1760 | T  | N  | **0.76**  | 0.62       |
| Deep   | 1760 | 32    | 1760 | T  | N  | **1.45**  | 1.39       |
| Deep   | 1760 | 64    | 1760 | T  | N  | **2.38**  | 2.36       |
| Deep   | 1760 | 128   | 1760 | T  | N  | **3.20**  | 2.46       |
| Deep   | 1760 | 7000  | 1760 | T  | N  | 6.89      | **8.94**   |
| Deep   | 2048 | 16    | 2048 | T  | N  | **0.99**  | 0.88       |
| Deep   | 2048 | 32    | 2048 | T  | N  | 1.97      | **2.18**   |
| Deep   | 2048 | 64    | 2048 | T  | N  | **2.77**  | 2.33       |
| Deep   | 2048 | 128   | 2048 | T  | N  | 3.08      | **4.50**   |
| Deep   | 2048 | 7000  | 2048 | T  | N  | **7.17**  | 6.89       |
| Deep   | 2560 | 16    | 2560 | T  | N  | **0.86**  | 0.71       |
| Deep   | 2560 | 32    | 2560 | T  | N  | 1.69      | **1.73**   |
| Deep   | 2560 | 64    | 2560 | T  | N  | **2.57**  | 2.49       |
| Deep   | 2560 | 128   | 2560 | T  | N  | 2.66      | **2.93**   |
| Deep   | 2560 | 7000  | 2560 | T  | N  | **7.03**  | 4.62       |
| Deep   | 1760 | 7133  | 1760 | N  | T  | 8.65      | **10.02**  |
| Deep   | 4096 | 7133  | 4096 | N  | T  | 8.98      | **9.73**   |
| Cov    | 32   | 60000 | 32   | N  | T  | **1.18**  | 0.74       |
| Cov    | 256  | 60000 | 256  | N  | T  | **6.78**  | 3.21       |
| Lapack | 4096 | 4096  | 32   | N  | T  | **4.17**  | 2.58       |
| Lapack | 3456 | 3456  | 32   | N  | T  | **3.90**  | 2.50       |
| Lapack | 896  | 896   | 32   | N  | T  | 1.25      | **1.34**   |

Cov	32	32	60000	N	T	1.18	0.74
Cov	256	256	60000	N	T	6.78	3.21
Lapack	4096	4096	32	N	T	4.17	2.58
Lapack	3456	3456	32	N	T	3.90	2.50
Lapack	896	896	32	N	T	1.25	1.34

### BLAS routines supported

Currently supported functions are:

| BLAS1         | BLAS2         | BLAS3         |
| --------------| --------------| --------------|
| xAXPY         | xGEMV         | xGEMM         |
| xCOPY         | xGER          |               |
| xSCAL         |               |               |
| xDOT          |               |               |
| xASUM         |               |               |

for x = {S, D}
