# ISAAC

This is the development branch for ISAAC v2.0. This is a major rewrite more targetted at compute-bound applications, with major performance gains at the expense of portability.

### License

ISAAC is distributed under the MIT/X11 license.

### Installation

ISAAC only requires an NVIDIA GPU with compute-capability > 5.0 and the corresponding proprietary driver. 

The CUDA SDK is *not* required.

```
git clone https://github.com/ptillet/isaac.git
mkdir -p isaac/build && cd isaac/build
cmake ../ && make -j8
./examples/bench
```

### Benchmarks
Below is the TFLOPS you get for sGEMM on a Pascal Titan X vs cuBLAS 8.0.
![alt tag](https://github.com/ptillet/isaac/raw/v2.0/documentation/bench/GEMM.png)

Below is the TFLOPS you get for FCONV on a Pascal Titan X vs cuDNN v6 [IMPLICIT_GEMM_PRECOMP].
![alt tag](https://github.com/ptillet/isaac/raw/v2.0/documentation/bench/CONV.png)

There's still room for improvement.

### APIs

ISAAC implements both GEMM and FCONV for fp16x2, fp32, and fp64. Half-precision with 32-bits accumulation and complex data-types is not yet supported.

### Future Plans

Future plans include (but are not limited to):
* Transparent use over cuBLAS/cuDNN using LD_PRELOAD
* Backward Convolution
* Complex data-types for GEMM
