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


unsigned char {binary_arg_name}[{bin_size}];
void load_{kernel_name}(void);
void unload_{kernel_name}(void);

// tt-linker: {kernel_name}:{signature}
CUresult {kernel_name}(CUstream stream, unsigned int gX,unsigned int gY,unsigned int gZ,unsigned int numWarps, {signature});

CUmodule {kernel_name}_mod = NULL;
CUfunction {kernel_name}_func = NULL;
unsigned char {binary_arg_name}[{bin_size}] = {{ {binary_val} }};


void unload_{kernel_name}(void) {{
    CUDA_CHECK(cuModuleUnload({kernel_name}_mod));
}}

void load_{kernel_name}(void) {{
    void *image = (void *)&{binary_arg_name};
    CUDA_CHECK(cuModuleLoadData(&{kernel_name}_mod, image));
    CUDA_CHECK(cuModuleGetFunction(&{kernel_name}_func, {kernel_name}_mod, "{compiled_func_name}"));
    // TODO: shared memory
}}

/*
{kernel_docstring}
*/
CUresult {kernel_name}(CUstream stream, unsigned int gX,unsigned int gY,unsigned int gZ,unsigned int numWarps, {signature}) {{
    if ({kernel_name}_func == NULL)
       load_{kernel_name}();
    void *args[{num_args}] = {{ {arg_pointers} }};
    // TODO: shared memory
    return cuLaunchKernel({kernel_name}_func, gX, gY, gZ, numWarps * {threads_per_warp}, 1, 1, 0, stream, args, NULL);
}}
