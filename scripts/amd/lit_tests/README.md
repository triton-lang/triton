# LIT test generator scripts

### generate_accelerate_matmul_tests.pp

This script generates CDNA related tests for AccelerateAMDMatmul pass.
There are 3 generations of CDNA architecture, so to generate all tests following commands are needed:

``` bash
python3 generate_accelerate_matmul_tests.py 2 ../../../test/TritonGPU/accelerate-matmul-cdna1.mlir
python3 generate_accelerate_matmul_tests.py 2 ../../../test/TritonGPU/accelerate-matmul-cdna2.mlir
python3 generate_accelerate_matmul_tests.py 2 ../../../test/TritonGPU/accelerate-matmul-cdna3.mlir
```

### generate_mfma_variants.py

This script generates CDNA related tests for TritonGPU to LLVM transformation:

``` bash
python3 generate_mfma_variants.py ../../../test/Conversion/AMDGPU/mfma_variants.mlir
```

