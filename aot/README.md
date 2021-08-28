# Ahead of Time Triton Compiler
This is a prototype for discussion. Not usable at all, only to ideate stucture.

The compiler takes a Triton kernel script and input type specification via config file as input and produces a shared library
that launches the kernels.

### Config sample
```toml
[named_type_variants]
K = "1, 8"
M = "1, 8"
N = "1, 8"
"*i" = "16,24,28,30,31,1"
A = "16,24"

[compile_config]
num_warps = 4
num_stages = 4
force_nc_cache = false

[kernels.add_kernel]
types = "f32*,f32*,f32*,i64"
type_variants = "_,_,_,A"

[kernels._matmul]
types = "f32*,f32*,f32*,i32,i32,i32,I,I,I,I,I,I"
type_variants = "_, _, _, A, A, A, M, K, K, N, M, N"

[kernels.add_kernel.meta]
BLOCK_SIZE = 1024

[kernels._matmul.meta]
BLOCK_M = 32
BLOCK_N = 64
BLOCK_K = 32
ACTIVATION = "leaky_relu"

```
### Output header example:
```c
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <cuda.h>


typedef struct
{
int gX;
int gY;
int gZ;
int numWarps;
} GridWarps;


unsigned char add_kernel_i64_16_ptx[6150];
CUfunction add_kernel_i64_16_fn();
CUresult add_kernel_i64_16(CUstream stream, GridWarps g, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int64_t n_elements);

```

## Whats going?

1. Load and executre Triton module ([tt_bindings](tt_bindings.py))
2. Parse config and build abstract inputy data ([compilation_config](compilation_config.py))
3. Compile kernels with Triton ([aot_kernel](aot_kernel.py))
4. C code generation ([c_codegen](c_codegen.py))
5. C compilation ([ttc](ttc.py))

### Noteable missing parts
- auto kernel selection based on input sizes
- compilation error handling

### Points to discuss
- best way to load kernels
- ptx? cubin?
