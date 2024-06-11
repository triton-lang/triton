#### Agenda:

##### Announcements:
1. Triton conference registration opening soon. Conference on 20th September at the Microsoft Silicon Valley Campus.

##### Items:
1. H100 updates
2. Triton release plan update
3. Linalg updates
4. Intel GPU Backend status update.
5. Intel working on the CPU backend for Triton.
6. AMD updates
7. Open discussion

##### Minutes:
Recording link [here](https://drive.google.com/file/d/19Nnc0i7zUyn-ni2RSFHbPHHiPkYU96Mz/view)

1. H100 updates:
   - Preliminary support is merged, disabled by default, can be enabled with env variables
   - Supports latest tensor cores, FP8s. Support for Flash Attention on the main branch coming soon.
   - Performance is very good on Matmuls, 80-90% of cublas on large Matmuls right now, will eventually reach parity with cublas. Above 600 teraflops on fp16 on xxm card, cublas is 670 on random input data. FP8 is twice that, around 1.2 petaflops.
   - Hopper support includes the full FP8 support for compute.
2. Triton release plan update
   - No specific dates for now, plan is to release before end of 2023.
   - Will move to 3.0 release due to minor backward compatibility breaking changes. For eg. Will move compiler options in the indexing operators as hardcoded operators in the kernel, will bump the major version.
   - Functionally the main goal will be to have 3rd party plugins for Intel and AMD gpus.
   - May synchronise with a PyTorch release so that PyTorch can benefit from the latest features, however continuous integration workflow is the default release cadence expected.
   - Will switch the default behavior to optimized mode for the release, needs more discussion with Nvidia.
   - Will expose flags for a user to enable kernel selection themselves.
   - Open question: Pytorch hasn’t rebased to latest triton, it is close to PyTorch code freeze – will PyTorch still sync with Triton 2.0? Will we have another release to support triton 2.0?
   - Community can start with the latest stable branch and rebase 3rd party plugin on top of that. OAI has no resources to commit to, but community can contribute.
3. Linalg updates
   - Discussion on Github for Linalg as a middle layer between the language and target hardware. Includes support for block pointers and modulo operators.
   - Please join the conversation [here](https://github.com/triton-lang/triton/discussions/1842)
   - Branch pushed is behind the tip, will work on getting it caught up on the tip.
4. Intel GPU Backend status update.
   - Please refer to slides [here](https://github.com/triton-lang/triton/blob/main/docs/meetups/Intel%20XPU%20Backend%20for%20Triton%20-%20Update%20-%200823.pptx)
5. Intel working on the CPU backend for Triton.
   - Please refer to slides [here](https://github.com/triton-lang/triton/blob/main/docs/meetups/Intel%20XPU%20Backend%20for%20Triton%20-%20Update%20-%200823.pptx)
6. AMD updates
   - Please refer to slides [here](https://github.com/triton-lang/triton/blob/main/docs/meetups/Triton_AMD_update_0823.pdf).
