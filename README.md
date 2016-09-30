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
where OP is axpy, gemv or gemm. It detects clBLAS or cuBLAS and compares it against ISAAC for DeepBench, Covariance, LAPACK (packed rank1 updates) and Square computations

Here's what you get for  on a Pascal Titan X (TFLOPS):

| BENCH | M    | N    | K    | AT | BT | ISAAC | cuBLAS |
| ------| -----| -----| -----| ---| ---| ------| -------|
| Deep  | 1760 | 16   | 1760 | N  | N  | 1.29  | 1.65   |





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
