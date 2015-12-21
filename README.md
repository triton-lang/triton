# ISAAC

This is the developer repository for ISAAC, a **modern** C++11 library for Numerical Computing. 
ISAAC uses Machine-Learning techniques to achieve input-specific and architecture-aware computations, thereby outperforming cuBLAS and clBLAS on many applications for NVidia, Intel and AMD GPUs.

**This is an ALPHA version**: only the C API is supported for now. 

ISAAC is dependency-free, and will load either OpenCL or CUDA 7.0+ _dynamically_ depending on which GPUs are detected at runtime. To install it, simply run:
```
git clone https://github.com/ptillet/isaac.git
mkdir -p isaac/build && cd isaac/build
cmake ../ && make -j4
```

The C API implements several binary symbols of both clBLAS and cuBLAS. To use ISAAC, simply link your application against libisaac.so instead of libclblas.so or libcublas.so

The C++ API relies on its own dynamic typing system and JIT compilation to achieve peak performance (i.e., auto-tuning, loop fusion and temporaries removal) all while preserving Numpy-like interface.

The Python API of ISAAC provides a tight wrapper around the C++ API.
