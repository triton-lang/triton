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

# Compatibility

Supported Platforms:
  * Linux

Supported Hardware:
  * NVIDIA GPUs (Compute Capability 7.0+)
  * Under development: AMD GPUs, CPUs

# Disclaimer

Triton is a fairly recent project, and it is under active development. We expect it to be pretty useful in a wide variety of cases, but don't be surprised if it's a bit rough around the edges :)