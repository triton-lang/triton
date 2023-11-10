from dataclasses import dataclass
from typing import Set


@dataclass
class AOTTemplate(dict):
    TEMPLATE_NAME: str
    PARAMS: Set[str]
    TEMPLATE: str

    def __post_init__(self):
        self.update(self.__dict__)


AOT_C_CUDA_Header_Template = AOTTemplate(
    TEMPLATE_NAME="C CUDA Kernel Header Template",
    PARAMS={
        "kernel_name",
        "full_signature",
        "algo_info",
        "signature",
        "_placeholder",
    },
    TEMPLATE="""
#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload_{kernel_name}(void);
void load_{kernel_name}(void);
// tt-linker: {kernel_name}:{full_signature}:{algo_info}
CUresult{_placeholder} {kernel_name}(CUstream stream, {signature});
""",
)

AOT_C_CUDA_Source_Template = AOTTemplate(
    TEMPLATE_NAME="C CUDA Kernel Source Template",
    PARAMS={
        "kernel_name",
        "bin_size",
        "bin_data",
        "shared",
        "triton_kernel_name",
        "kernel_docstring",
        "signature",
        "gridX",
        "gridY",
        "gridZ",
        "num_args",
        "arg_pointers",
        "num_warps",
    },
    TEMPLATE="""
/* clang-format off */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <cuda.h>


// helpers to check for cuda errors
#define CUDA_CHECK(ans) {{\\
    gpuAssert((ans), __FILE__, __LINE__);\\
  }}\\

static inline void gpuAssert(CUresult code, const char *file, int line) {{
  if (code != CUDA_SUCCESS) {{
    const char *prefix = "Triton Error [CUDA]: ";
    const char *str;
    cuGetErrorString(code, &str);
    char err[1024] = {{0}};
    strcat(err, prefix);
    strcat(err, str);
    printf("%s\\\\n", err);
    exit(code);
  }}
}}

// globals
#define CUBIN_NAME {kernel_name}_cubin
CUmodule {kernel_name}_mod = NULL;
CUfunction {kernel_name}_func = NULL;
unsigned char CUBIN_NAME[{bin_size}] = {{ {bin_data} }};


void unload_{kernel_name}(void) {{
    CUDA_CHECK(cuModuleUnload({kernel_name}_mod));
}}

// TODO: some code duplication with `runtime/backend/cuda.c`
void load_{kernel_name}() {{
    int dev = 0;
    void *bin = (void *)&CUBIN_NAME;
    int shared = {shared};
    CUDA_CHECK(cuModuleLoadData(&{kernel_name}_mod, bin));
    CUDA_CHECK(cuModuleGetFunction(&{kernel_name}_func, {kernel_name}_mod, "{triton_kernel_name}"));
    // set dynamic shared memory if necessary
    int shared_optin;
    CUDA_CHECK(cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
    if (shared > 49152 && shared_optin > 49152) {{
      CUDA_CHECK(cuFuncSetCacheConfig({kernel_name}_func, CU_FUNC_CACHE_PREFER_SHARED));
      CUDA_CHECK(cuFuncSetAttribute({kernel_name}_func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin))
    }}
}}

/*
{kernel_docstring}
*/
CUresult {kernel_name}(CUstream stream, {signature}) {{
    if ({kernel_name}_func == NULL)
       load_{kernel_name}();
    unsigned int gX = {gridX};
    unsigned int gY = {gridY};
    unsigned int gZ = {gridZ};
    void *args[{num_args}] = {{ {arg_pointers} }};
    // TODO: shared memory
    if(gX * gY * gZ > 0)
      return cuLaunchKernel({kernel_name}_func, gX, gY, gZ, {num_warps} * 32, 1, 1, {shared}, stream, args, NULL);
}}
""",
)

DEFAULT_AOT_C_CUDA_HEADER_TEMPLATE = AOT_C_CUDA_Header_Template
DEFAULT_AOT_C_CUDA_SOURCE_TEMPLATE = AOT_C_CUDA_Source_Template
