# Triton Programming Language Contributor's Guide

First of all, thank you for considering contributing to the Triton programming language! We appreciate the time and effort you're willing to put into improving and expanding our project. In order to maintain a high standard of code and a welcoming atmosphere for collaboration, we kindly ask you to follow the guidelines outlined below.

## General Guidelines

1. **Quality Contributions:** We value meaningful contributions that aim to improve the project and help it grow. Please refrain from submitting low-effort pull requests (PR) -- such as minor formatting/typo fixes -- solely for the purpose of appearing in the commit history. Maintainers have limited bandwidth, and may decline to review such work.

2. **Code Formatting:** Our Continuous Integration (CI) pipeline uses autopep8, isort, and clang-format to check code formatting. To avoid failing the CI workflow due to formatting issues, please utilize the provided `.pre-commit-config.yaml` pre-commit configuration file.

3. **Unit Testing:** When contributing new functionalities, please also include appropriate tests. We aim to continuously improve and expand our CI pipeline to ensure the robustness and reliability of the project. PRs that add a large amount of untested code will be rejected.

4. **Respectful Communication:** In all discussions related to PRs or other contributions, please maintain a courteous and civil tone. We strive to foster a collaborative environment that is inclusive and respectful to all contributors.


## Request for Comments (RFCs)

RFCs are a crucial aspect of the collaborative development process, as they provide a structured way to propose and discuss significant changes or additions to the project. RFCs may encompass modifications to the language itself, extensive changes in the compiler backend, or other substantial updates that impact the Triton ecosystem.

To ensure that RFCs are clear and easy to understand, consider the following guidelines when creating one:

### Purpose

The purpose of an RFC is to:

- Clearly communicate your proposal to the Triton community
- Collect feedback from maintainers and other contributors
- Provide a platform for discussing and refining ideas
- Reach a consensus on the best approach for implementing the proposed changes

### Structure

A well-structured RFC should include:

1. **Title:** A concise and descriptive title that reflects the main topic of the proposal.

2. **Summary:** A brief overview of the proposed changes, including the motivation behind them and their intended impact on the project.

3. **Detailed Design:** A thorough description of the proposed changes, including:
   - Technical details and implementation approach
   - Any new or modified components, functions, or data structures
   - Any potential challenges or limitations, as well as proposed solutions

4. **Examples and Use Cases:** Provide examples of how the proposed changes would be used in real-world scenarios, as well as any use cases that demonstrate the benefits of the changes.

5. **Performance Impact:** Discuss the expected performance impact of the proposed changes, including any potential bottlenecks or performance improvements.

6. **Timeline and Milestones:** Outline a proposed timeline for implementing the changes, including any milestones or intermediate steps.


## New backends

Due to limited resources, we need to prioritize the number of targets we support. We are committed to providing upstream support for Nvidia and AMD GPUs. However, if you wish to contribute support for other backends, please start your project in a fork. If your backend proves to be useful and meets our performance requirements, we will discuss the possibility of upstreaming it.


## Project Structure
```
triton
├── lib : C++ code for python library
│   ├──Analysis
│   │	Memory barrier analysis
│   │	class to extract axis information from MLIR ops
│   │	implementation of the shared memory allocation analysis for Triton dialect
│   │
│   ├──Conversion
│   │	├──TritonGPUToLLVM:  Transforms TritonGPU  to LLVM;
│   │	│
│   │	├──TritonToTritonGPU: Transforms ops to TritonGPU ops; loading, storing, arithmetic, casting, and tensor operations.
│   │	│
│   │	│
│   │	│
│   ├──Dialect
│   │	├──Triton
│   │	│	Defines core IR for Triton compiler
│   │	├──TritonGPU
│   │	    Defines TritonGPU operation for IR
│   │
│   ├──Target: contains Triton targets for converting to PTX, LLVMIR and HSACO IR targets
│   │
├── bin
├── cmake
├── docs ├── Documentation regarding using triton
├── include
│   CMakelists.txt
│   ├──triton
│   │   ├──
├── python
│   ├──
│   ├── MANIFEST.in
│   ├── README.md
│   ├── build
│   ├── examples
│   ├── pyproject.toml
│   ├── setup.py: pip install for python package
│   ├── src
│   ├── test
│   ├── triton
│   │	├── _C: Includes header files and compiled .so file for C library
│   │	│
│   │	├──common: Has interface for CUDA hardware backend
│   │	│
│   │	├──compiler: contains code for compiling source code to IR and launching GPU kernels
│   │	│
│   │	├──interpreter: memory-map for tensors, converting primitives to tensors, and arethmetic ops for tensors
│   │	│
│   │	├──language: core of triton language, load tensors to SRAM, language logic, etc.
│   │	│
│   │	├──ops: contains functions for flash-attn, softmax, cross-entropy and other torch.nn.F functions
│   │	├──runtime: contains impl jit compilation, autotuning, backend drivers, caching, error handles, etc.
│   │	├──third_party
│   │	├──tools
│   ├── triton.egg-info
│   ├── tutorials: contains tutorials for various use-cases
├── test
│   ├──Analysis
│   ├──Conversion
│   ├──Dialect
│   ├──Target
├── third_party
├── unittest
└── utils
```
