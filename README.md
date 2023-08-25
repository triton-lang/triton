<div align="center">
  <img src="https://cdn.openai.com/triton/assets/triton-logo.png" alt="Triton logo" width="88" height="100">
</div>

[![Wheels](https://github.com/openai/triton/actions/workflows/wheels.yml/badge.svg?branch=release/2.0.x)](https://github.com/openai/triton/actions/workflows/wheels.yml)


**`Documentation`** |
------------------- |
[![Documentation](https://github.com/openai/triton/actions/workflows/documentation.yml/badge.svg)](https://triton-lang.org/)

# Triton Developer Conference Registration Open
The Triton Developer Conference will be held in a hybrid mode at the Microsoft Silicon Valley Campus in Mountain View, California. The conference will be held on September 20th from 10am to 4pm, followed by a reception till 5:30 pm. Please use the link below to register to attend either in-person or virtually online.

Registration Link for Triton Developer Conference is [here](https://forms.office.com/r/m4jQXShDts)

Tentative Agenda for the conference (subject to change):

|Time    |Title  |Speaker
|--------|-------|-------|
|10:00 AM|Welcome|Kevin Scott (Microsoft)|
|10:20 AM|The Triton Compiler: Past, Present and Future|Phil Tillet (OpenAI)|
|11:00 AM|**Break**||
|11:20 AM|Hopper support in Triton|Gustav Zhu (Nvidia)|
|11:40 AM|Bringing Triton to AMD GPUs|Jason Furmanek, Lixun Zhang (AMD)|
|12:00 PM|Intel XPU Backend for Triton|Eikan Wang (Intel)|
|12:20 PM|Vectorization of Triton Kernels for Qualcomm Hexagon Backend|Javed Absar (Qualcomm)|
|12:30 PM|**Lunch**||
|1:40 PM |Triton for MTIA|Roman Levenstein et al, (Meta)|
|2:00 PM |Using Triton IR for high-performance fusions in XLA|George Karpenkov (Google)|
|2:20 PM |Triton for All: Triton as a device-independent language|Ian Bearman (Microsoft)|
|2:40 PM|**Break**||
|3:00 PM|PyTorch 2.0 and TorchInductor|Jason Ansel, Horace He (Meta)|
|3:20 PM|Pallas: A JAX Kernel Language|Sharad Vikram (Google)|
|3:40 PM|Writing Grouped GEMMs in Triton|Vinod Grover (Nvidia)|
|4:00 PM|**Reception**||


# Triton

This is the development repository of Triton, a language and compiler for writing highly efficient custom Deep-Learning primitives. The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs.

The foundations of this project are described in the following MAPL2019 publication: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). Please consider citing this work if you use Triton!

The [official documentation](https://triton-lang.org) contains installation instructions and tutorials.

# Quick Installation

You can install the latest stable release of Triton from pip:

```bash
pip install triton
```
Binary wheels are available for CPython 3.7-3.11 and PyPy 3.8-3.9.

And the latest nightly release:

```bash
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

# Install from source

```
git clone https://github.com/openai/triton.git;
cd triton/python;
pip install cmake; # build-time dependency
pip install -e .
```



# Changelog

Version 2.0 is out! New features include:
- Many, many bug fixes
- Performance improvements
- Backend rewritten to use MLIR
- Support for kernels that contain back-to-back matmuls (e.g., flash attention)

# Contributing

Community contributions are more than welcome, whether it be to fix bugs or to add new features at [github](https://github.com/openai/triton/). For more detailed instructions, please visit our [contributor's guide](CONTRIBUTING.md).

If you’re interested in joining our team and working on Triton & GPU kernels, [we’re hiring](https://openai.com/jobs/#acceleration)!




# Compatibility

Supported Platforms:
  * Linux

Supported Hardware:
  * NVIDIA GPUs (Compute Capability 7.0+)
  * Under development: AMD GPUs, CPUs
