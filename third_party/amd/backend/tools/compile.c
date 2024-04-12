#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <stdlib.h>


// helpers to check for hip errors
#define HIP_CHECK(ans) {{\
    gpuAssert((ans), __FILE__, __LINE__);\
  }}\

static inline void gpuAssert(hipError_t code, const char *file, int line) {{
  if (code != HIP_SUCCESS) {{
    const char *prefix = "Triton Error [HIP]: ";
    const char* str = hipGetErrorString(code);
    char err[1024] = {{0}};
    strcat(err, prefix);
    strcat(err, str);
    printf("%s\\n", err);
    exit(code);
  }}
}}

// globals
#define HSACO_NAME {kernel_name}_hsaco
hipModule_t {kernel_name}_mod = NULL;
hipFunction_t {kernel_name}_func = NULL;
unsigned char HSACO_NAME[{bin_size}] = {{ {bin_data} }};

void unload_{kernel_name}(void) {{
    HIP_CHECK(hipModuleUnload({kernel_name}_mod));
}}


// TODO: some code duplication with `third_party/amd/backend/driver.c`
void load_{kernel_name}() {{
    int dev = 0;
    void *bin = (void *)&HSACO_NAME;
    int shared = {shared};

    HIP_CHECK(hipModuleLoadDataEx(&{kernel_name}_mod, bin, 5, 0, 0));
    HIP_CHECK(hipModuleGetFunction(&{kernel_name}_func, {kernel_name}_mod, "{triton_kernel_name}"));
}}

/*
{kernel_docstring}
*/
hipError_t {kernel_name}(hipStream_t stream, {signature}) {{
    if ({kernel_name}_func == NULL)
       load_{kernel_name}();
    unsigned int gX = {gridX};
    unsigned int gY = {gridY};
    unsigned int gZ = {gridZ};
    void *args[{num_args}] = {{ {arg_pointers} }};
    // TODO: shared memory
    if(gX * gY * gZ > 0)
      return (hipModuleLaunchKernel({kernel_name}_func, gX, gY, gZ, {num_warps} * 64, 1, 1, {shared}, stream, args, NULL));
}}
