# ISAAC

This is the developer repository for ISAAC, a library that uses machine learning to find input-aware kernels for element-wise operations, 1D/2D reductions and GEMM. It works with both cuBLAS and clBLAS, and fallbacks on those when appropriate (typically large square matrices).

### License

ISAAC is distributed under the MIT License.

### Installation on Linux

ISAAC is dependency-free, and will load either OpenCL and/or CUDA 7.0+ _dynamically_ depending on which GPUs are detected at runtime.

You only need CMake 2.8.7+ and a C++11 compliant compiler:  
 

```
git clone https://github.com/ptillet/isaac.git
mkdir -p isaac/build && cd isaac/build
cmake ../ && make -j4
```

Link against libisaac.so instead of libcublas.so or libclblas.so, and you're good to go! 

The C++ and Python API does some kernel fusion, but is not entirely stable. It works well to compose element-wise operations, though.

### Installation on Windows
The windows version depends on dlfcn project, please download it at **[dlfcn windows version](https://github.com/dlfcn-win32/dlfcn-win32)** and save it at the same directory as isaac project, build ***dlfcn*** before build isaac.
The following instructions are for Visual C 2015 and to build it for x64 target.


```
git clone https://github.com/dlfcn-win32/dlfcn-win32
cd dlfcn-win32
cmake -G "Visual Studio 14 2015 Win64" .
cmake --build . --config Release
git clone https://github.com/ptillet/isaac.git
mkdir -p isaac/build
cd build
cmake -G "Visual Studio 14 2015 Win64" ..
cmake --build . --config Release

```

### Benchmark

```
Usage : blas-bench [--op {axpy, dot, gemv, gemm}] [--dtype {float32, float64}] [--device DEVICE_IDX] [--help]
--op: operation to benchmark (default = gemm)
--dtype: data-type to benchmark (default = float32)
--device: index of isaac device in [0, ..., ndevices - 1] (default = 0)
--help: display this message
```
It detects clBLAS or cuBLAS and compares it against ISAAC for e.g., DeepBench, Covariance, LAPACK (packed rank1 updates), etc.

Below is the TFLOPS you get for GEMM on a Pascal Titan X (cuBLAS 8.0). Numbers in bold represent speed-ups greater than 5%.
![alt tag](https://github.com/ptillet/isaac/raw/master/documentation/bench/bench-cuBLAS.png)

For AMD Fury (clBLAS-2.10-Fiji):
![alt tag](https://github.com/ptillet/isaac/raw/master/documentation/bench/bench-clBLAS.png)

Same trend on Intel Broadwell iGPU

### BLAS routines supported

Currently supported functions are:

| BLAS1         | BLAS2         | BLAS3         |
| --------------| --------------| --------------|
| xAXPY         | xGEMV         | xGEMM         |
| xCOPY         | xGER          |               |
| xSCAL         |               |               |
| xDOT          |               |               |
| xASUM         |               |               |

For x in {S, D}

### Contributing

You can contribute to further tuning isaac if you have one of the following architecture:
- NVidia: SM 2.x ; SM 3.5 ; SM 5.0

If you have one of the following architectures you can contribute by running:

```
git clone https://github.com/ptillet/isaac.git
cd isaac/python ;
python setup.py build;
cd ../tune
PYTHONPATH=../python/build/lib.linux-x86_64-2.7/ python main.py --float64 --float32 --elementwise_1d --elementwise_2d --reduce_1d --reduce_2d_rows --reduce_2d_cols --gemm_nn --gemm_nt --gemm_tn --gemm_tt
```

This will output a .json file that you can submit for integration.

Bug reports are more than welcome!
