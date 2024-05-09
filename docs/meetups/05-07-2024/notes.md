#### Agenda:
1. Triton CPU summary
2. Triton introduced a new Triton layout redesign (linear layout PR3794 ). Does this layout try to cover Triton CPU backend for SIMD instructions.
3. Triton Stream-k on AMD GPUs

##### Items:
Meeting notes:
1. Triton CPU backend: The Meta team presented their motivation, design, and progress on developing a CPU backend for Triton.
   There is a demand for heterogeneity and portability across different CPU architectures, especially for small batch sizes and inference workloads.
   They proposed to use MLIR and vector dialect to lower Triton IR to LLVM IR, and to leverage existing dialects and transformations for GPU backends.
   There maybe a possible refactoring of the CPU backend to make it more general and modular.
   Currently they have done initial work on plumbing the CPU backend and implementing a basic vector load operation using transfer read.
   Repo and other details are in the slides below.
   Open questions: How to handle different vector widths and operations, how to support ARM Neon, how to set performance goals and criteria, and how to coordinate with other Triton developers and contributors.
2. Stream-k for AMD: The AMD team presented their implementation and evaluation of Stream-k, a load-balanced scheme for matrix multiplication that can handle different tile sizes and split K dimensions.
   They compared it with PyTorch Matmul and Triton Matmul. Other details are in the slides below.

##### Minutes:
Recording link [here](https://youtu.be/hgINpebZ7n0)

Presentations repo [here](https://drive.google.com/drive/folders/1xPnRO5P59aMVJnXz_o9ASTUgTXK1lhHW?usp=drive_link)
