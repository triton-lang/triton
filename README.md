<img src="https://cdn.openai.com/triton/assets/triton-logo.png" alt="Triton logo" width="80" height="91">

# Triton

This is the development repository of Triton, a language and compiler for writing highly efficient custom Deep-Learning primitives. The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs.

[![Build Status](https://dev.azure.com/triton-lang/Triton/_apis/build/status/ptillet.triton?branchName=master)](https://dev.azure.com/triton-lang/Triton/_build/latest?definitionId=10&branchName=master)

The foundations of this project are described in the following MAPL2019 publication: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). Please consider citing us if you use our work!

The [official documentation](https://triton-lang.org) contains installation instructions and tutorials.

# Compatibility

Supported Platforms:
  * Linux

Supported Hardware:
  * NVIDIA GPUs (Compute Capability 7.0+)
  * Under development: AMD GPUs, CPUs
