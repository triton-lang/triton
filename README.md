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

# Changelog

Version 1.1 is out! New features include:
- Many, many bugfixes
- More documentation
- Automatic on-disk caching of compiled binary objects
- Random Number Generation
- Faster (up to 2x on A100), cleaner blocksparse ops

# Contributing

Community contributions are more than welcome, whether it be to fix bugs or to add new features. Feel free to open GitHub issues about your contribution ideas, and we will review them. A contributor's guide containing general guidelines is coming soon!

If you’re interested in joining our team and working on Triton & GPU kernels, [we’re hiring](https://openai.com/jobs/#acceleration)!


# Compatibility

Supported Platforms:
  * Linux

Supported Hardware:
  * NVIDIA GPUs (Compute Capability 7.0+)
  * Under development: AMD GPUs, CPUs

# Disclaimer

Triton is a fairly recent project, and it is under active development. We expect it to be pretty useful in a wide variety of cases, but don't be surprised if it's a bit rough around the edges :)
