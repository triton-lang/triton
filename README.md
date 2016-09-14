# ISAAC

This is the developer repository for ISAAC, a new approach to BLAS implementations. 
ISAAC uses Machine-Learning techniques to achieve input-specific and architecture-aware computations, thereby outperforming cuBLAS and clBLAS on many applications for NVidia, Intel and AMD GPUs.

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
where OP is axpy, gemv or gemm

If you have multiple devices

```
./bench/bench-blas DEVICE_ID OP
```

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
