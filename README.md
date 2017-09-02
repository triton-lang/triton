# ISAAC

This is the development branch for ISAAC v2.0. This is a major rewrite more targetted at compute-bound applications, with major performance gains at the expense of portability.

Major changes compared to the master branch are:
* PTX code generation for GEMM and FCONV
* Double-throughput half-precision via the FFMA.F16x2 instruction
* Much faster auto-tuning procedure: obtaining an input-aware GEMM profile for all layouts/data-types shouldn't take more than 3 hours.
* Massive code simplifications: compiling the entire project shouldn't take more than 20 seconds


### License

ISAAC is distributed under the MIT/X11 license.

### Installation

ISAAC only needs an NVIDIA GPU with compute-capability > 5.0 and the corresponding proprietary driver. 

The CUDA SDK is *not* required.

```
git clone -b v2.0 https://github.com/ptillet/isaac.git
mkdir -p isaac/build && cd isaac/build
cmake ../ && make -j8
optirun ./examples/bench
```

### Benchmarks
Below is the TFLOPS you get for sGEMM on a Pascal Titan X vs cuBLAS 8.0.
![alt tag](https://github.com/ptillet/isaac/raw/v2.0/documentation/bench/GEMM.png)

Below is the TFLOPS you get for FCONV on a Pascal Titan X vs cuDNN v6 [IMPLICIT_GEMM_PRECOMP].
![alt tag](https://github.com/ptillet/isaac/raw/v2.0/documentation/bench/CONV.png)

### Future Plans
Future plans include:
* Transparent use over cuBLAS/cuDNN using LD_PRELOAD
* Backward Convolution
* Complex data-types for GEMM
* 32-bits accumulations for hCONV
