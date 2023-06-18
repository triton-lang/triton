/*[ kernel_header ]*/
/* clang-format off */

#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <cuda.h>

#endif

static inline void gpuAssert(CUresult code, const char *file, int line) {{
  if (code != CUDA_SUCCESS) {{
    const char *prefix = "Triton Error [CUDA]: ";
    const char *str;
    cuGetErrorString(code, &str);
    char err[1024] = {{0}};
    strcat(err, prefix);
    strcat(err, str);
    printf("%s\\n", err);
    exit(code);
  }}
}}

#define CUDA_CHECK(ans) {{
    gpuAssert((ans), __FILE__, __LINE__);
  }}

#define CUBIN_NAME {kernel_name}_cubin

unsigned char CUBIN_NAME[{bin_size}];
void load_{kernel_name}(void);
void unload_{kernel_name}(void);

// tt-linker: {kernel_name}:{signature}
CUresult {kernel_name}(CUstream stream, unsigned int gX,unsigned int gY,unsigned int gZ,unsigned int numWarps, {signature});

CUmodule {kernel_name}_mod = NULL;
CUfunction {kernel_name}_func = NULL;
unsigned char CUBIN_NAME[{bin_size}] = {{ {binary_val} }};


void unload_{kernel_name}(void) {{
    CUDA_CHECK(cuModuleUnload({kernel_name}_mod));
}}

// TODO: some code duplication with `runtime/backend/cuda.c`
void load_{kernel_name}(int dev = 0) {{
    void *bin = (void *)&CUBIN_NAME;
    int shared = {shared};
    CUDA_CHECK(cuModuleLoadData(&{kernel_name}_mod, bin));
    CUDA_CHECK(cuModuleGetFunction(&{kernel_name}_func, {kernel_name}_mod, "{kernel_name}"));
    // set dynamic shared memory if necessary
    int shared_optin;
    CUDA_CHECK(cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
    if (shared > 49152 && shared_optin > 49152) {{
      CUDA_CHECK(cuFuncSetCacheConfig(fun, CU_FUNC_CACHE_PREFER_SHARED));
      CUDA_CHECK(cuFuncSetAttribute(fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin))
    }}
}}

/*
{kernel_docstring}
*/
CUresult {kernel_name}(CUstream stream, unsigned int gX,unsigned int gY,unsigned int gZ,unsigned int numWarps, {signature}) {{
    if ({kernel_name}_func == NULL)
       load_{kernel_name}();
    void *args[{num_args}] = {{ {arg_pointers} }};
    // TODO: shared memory
    if(gX * gY * gZ > 0)
      return cuLaunchKernel({kernel_name}_func, gX, gY, gZ, numWarps * 32, 1, 1, {shared}, stream, args, NULL);
}}
