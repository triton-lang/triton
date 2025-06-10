#### Agenda:
1. Improving ILP (instruction level parallelism) with Warp Specialization - Hongtao Yu (Meta), Manman Ren (Meta), Yuanwei (Kevin) Fang (Meta)
2. AMD Updates
3. Shared Middle Layer Updates
4. Intel Questions

##### Items:
Meeting notes:
1. Warp specialization
  - available on PyTorch 2.6, and triton-lang/release 3.2.x.
  - ILP - Instruction Level Parallelism.
  - Includes case study on Flash Attention.
  - Meta has FP8 GEMM example.  8 warp GEMM, each warp does 1/8th of work.
  - Separate problem into: warp scopes 
    - 4 warps to produce data (loads)
    - 2 groups of 4 warps to consume data (MMAs until they run out of data)
  - Enable by adding 
2. Intel GPU Backend: Intel GPU backend shows good performance close to expert-tuned kernels and the use of block pointers for performance gains. There were questions around the future of block pointers and their importance for performance gains. With block-pointer deprecation there is a need for a more generic interface to support various backends including Intel GPU.
3. The 2024 Triton conference is on September 17th 2024 in Fremont California! Please register [here](README.md).
##### Minutes:
Recording link [here](https://youtu.be/dfL3L4_3ujg)

Presentations repo [here](https://drive.google.com/drive/folders/1fQ3zVrM7DT8W8FGJWKx1wNr2X53tYbeT?usp=sharing)
