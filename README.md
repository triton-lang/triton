<div align="center">
  <img src="https://cdn.openai.com/triton/assets/triton-logo.png" alt="Triton logo" width="88" height="100">
</div>

[![Wheels](https://github.com/openai/triton/actions/workflows/wheels.yml/badge.svg)](https://github.com/openai/triton/actions/workflows/wheels.yml)


**`Documentation`** |
------------------- |
[![Documentation](https://github.com/openai/triton/actions/workflows/documentation.yml/badge.svg)](https://triton-lang.org/)


# Triton

This is the development repository of Triton, a language and compiler for writing highly efficient custom Deep-Learning primitives. The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs.

The foundations of this project are described in the following MAPL2019 publication: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). Please consider citing this work if you use Triton!

The [official documentation](https://triton-lang.org) contains installation instructions and tutorials.

# Quick Installation

You can install the latest stable release of Triton from pip:

```bash
pip install triton
```
Binary wheels are available for CPython 3.6-3.11 and PyPy 3.7-3.9.

And the latest nightly release:

```bash
pip install -U --pre triton
```

# Install from source

```
git clone https://github.com/openai/triton.git;
cd triton/python;
pip install cmake; # build time dependency
pip install -e .
```

# Changelog

Version 2.0 is out! New features include:
- Many, many bugfixes
- Performance improvements
- Backend rewritten to use MLIR
- Support for kernels taht contain back-to-back matmuls (e.g., flash attention)

# Contributing

Community contributions are more than welcome, whether it be to fix bugs or to add new features. Feel free to open GitHub issues about your contribution ideas, and we will review them. A contributor's guide containing general guidelines is coming soon!

If you’re interested in joining our team and working on Triton & GPU kernels, [we’re hiring](https://openai.com/jobs/#acceleration)!


# Compatibility

Supported Platforms:
  * Linux

Supported Hardware:
  * NVIDIA GPUs (Compute Capability 7.0+)
  * Under development: AMD GPUs, CPUs