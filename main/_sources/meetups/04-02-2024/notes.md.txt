#### Agenda:

##### Items:
1. Interpreter update
2. Experience with TMA support and future plans for it
3. CGO trip report
4. Triton upstream CI and unit test status from AMD
5. Open discussion

##### Minutes:
Recording link [here](https://youtu.be/VTcFe2XxZZc)

Presentations repo [here](https://drive.google.com/drive/folders/1bKpvz1NiBL_fHrGhMoZPvQfXCeetV2iY?usp=sharing)

1. Triton interpreter mode: The Open AI presented the interpreter mode for Triton code, which allows users to debug and inspect individual GPU programs using native Python print or PDB. It is currently being turned on using an environment variables, code decorators for individual functions being interpreted are still TBD. It can also run on CPU without GPU. For more details about the presentation please refer slides.
2. Tensor Memory Access (TMA) discussion: The current implementation of TMA in Triton has some limitations, so has been removed for now. The plan is to rethink how to do it better in the future. The goal is to support TMA implicitly, but the challenge is to handle the different memory layouts for different backends. There is a pull request to improve the launch overhead of kernels, which is related to TMA, but it would require extensive review and testing.
3. CGO trip report: Ian Bearman from Microsoft shared his experience of attending CGO and the Compilers for Machine Learning workshop. He and Javed Absar from Qualcomm gave talks about Triton shared and answered questions about Triton. There was a lot of interest in Triton as a cross-platform kernel language and questions were around the PyTorch integration, the performance portability, and the codegen bugs. It will be good to make the Triton-Pytorch connection more visible. There was also another project called Turbine that was similar to Triton. Please refer to the slides for more details.
4. AMD upstream CI and unit tests status: The AMD team discussed CI and enabling tests for MI 210 and MI 300. Work is in progress for performance gaps, compilation errors and fixes for FP8IN and flash attention kernels. The plan is to upstream these changes soon. Please refer to the slides for more details.
5. Third party CPU backend: The Intel team is driving discussions for community collaboration on a proof of concept for a CPU backend for Triton, using MLIR and OpenMP. There will be a follow-up meeting to discuss the logistics and design. Please refer to the third-party channel in slack for more details.
