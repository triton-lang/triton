# Triton

This is the development repository of Triton, a language and compiler for writing highly efficient custom Deep-Learning primitives. The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with much higher flexibility than [TVM](https://github.com/apache/incubator-tvm) and without having to manually specify compute schedules.

The main scope of Triton at the moment are:
- **Triton-C**: An imperative, single-threaded language for writing highly efficient compute-kernels at a relatively high abstraction level using numpy-like extensions of the C language.
- **Triton-IR**: An intermediate-representation for optimizing multi-dimensional array operations in linear algebra programs
- **Triton-JIT**: An optimizing just-in-time compiler for Triton-C, which generates GPU code on par with state-of-the-art CUDA-C  (e.g.,  [CUTLASS](https://github.com/NVIDIA/cutlass)) and PTX (e.g., [ISAAC](https://github.com/ptillet/isaac)). This includes transparent support for mixed-precision and Tensor Cores.

Bindings for **automatic** PyTorch custom op generations are included in -  **PyTriton**, along with a small DSL based on einsum that supports convolutions, shift-convolutions, direct einsums, etc.

The formal foundations of this project are described in the following MAPL2019 publication: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). Please cite us if you use our work!




## Installation

Triton is a fairly self-contained package and uses its own parser (forked from [wgtcc](https://github.com/wgtdkp/wgtcc)) and LLVM code-generator. However, at the moment it relies on LLVM-8.0+ for PTX code generation. The whole compiler stack (~30k lines of C++ code) should take around 15 secs to compile.

```
sudo apt-get install llvm-8-dev
git clone https://github.com/ptillet/triton.git;
cd triton/python/;
python setup.py develop;
cd examples;
python einsum.py
```

## Tutorials

- [The Triton-C language](https://github.com/ptillet/triton/blob/master/docs/triton-c.md)
- [The PyTriton API](https://github.com/ptillet/triton/blob/master/docs/pytriton.md)
- Extended Einstein Summations (coming soon...)
- The Triton-IR representation (coming soon...)
- The Triton-JIT compiler (coming soon...)

