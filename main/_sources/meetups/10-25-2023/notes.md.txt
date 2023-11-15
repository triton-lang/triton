#### Agenda:

##### Items:
1. H100 updates
2. Triton-Shared layer updates
3. Intel update
4. Open discussion

##### Minutes:
Recording link [here](https://youtu.be/KZAzpKx1ebI)

1. H100 updates
   - Enabled WGMMA by default, now any matmul can reuse it.
   - fp8 formats enabled – 1.3 Petaflops on dense matmul on H100 (gemm performance)
   - Enabled Flash Attention using wgmma, resulting in 450 teraflop on fwd pass and 250 on backward pass – still working on perf for flash attention
   - fp8 numbers with flash attention running in fp8 with matmul is tricky, because the fp8 layout is significantly different than what is returned by wgmma, still wip

2. Triton-Shared layer
   - Please refer to slides for more details
   - Created a repo where you can find the middle layer
   - Available as a plugin into triton

3. Intel Update
   - Please refer to slides for more details
