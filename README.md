# Triton

This is the development repository of Triton, a language and compiler for writing highly efficient custom Deep-Learning primitives. 

The formal foundations of this project are described in the following MAPL2019 publication: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). Please cite us if you use our work!


The main features of Triton at the moment are:
-  **PyTriton**: A Python API for writing custom operations for Triton-C compute-kernels. PyTriton automatically generates and just-in-time Tensorflow and PyTorch bindings.
- **Triton-C**: An imperative, single-threaded language for writing highly efficient compute-kernels at a relatively high abstraction level using numpy-like extensions of the C language.
- **Triton-IR**: An intermediate-representation for optimizing multi-dimensional array operations in linear algebra programs
- **Triton-JIT**: An optimizing just-in-time compiler for Triton-C, which generates GPU code on par with state-of-the-art CUDA-C  (e.g.,  [CUTLASS](https://github.com/NVIDIA/cutlass)) and PTX (e.g., [ISAAC](https://github.com/ptillet/isaac)). This includes transparent support for mixed-precision and Tensor Cores.




## Installation

Triton is a fairly self-contained package and uses its own parser (forked from [wgtcc](https://github.com/wgtdkp/wgtcc)) and LLVM code-generator. However, at the moment it still relies on LLVM-8.0+ for PTX code generation.

```
sudo apt-get install llvm-8-dev
git clone https://github.com/ptillet/triton.git;
cd triton/python/;
python setup.py develop;
cd examples;
python dot.py
```

## Tutorials

- [The Triton-C language](https://github.com/ptillet/triton/blob/master/docs/triton-c.md)
- [The PyTriton API](https://github.com/ptillet/triton/blob/master/docs/pytriton.md)
- The Triton-IR representation (coming soon...)
- The Triton-JIT compiler (coming soon...)

