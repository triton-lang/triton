// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/* clang-format off */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <hip/hip_runtime.h>

// helpers to check for hip errors
#define HIP_CHECK(ans) {{\
    gpuAssert((ans), __FILE__, __LINE__);\
  }}\

static inline void gpuAssert(hipError_t code, const char *file, int line) {{
  if (code != hipSuccess) {{
    const char *prefix = "Triton Error [HIP]: ";
    const char *str;
    hipDrvGetErrorString(code, &str);
    char err[1024] = {{0}};
    strcat(err, prefix);
    strcat(err, str);
    printf("%s\\n", err);
    exit(code);
  }}
}}

// globals
#define HSACO_NAME {kernel_name}_hsaco
hipModule_t {kernel_name}_mod = nullptr;
hipFunction_t {kernel_name}_func = nullptr;
unsigned char HSACO_NAME[{bin_size}] = {{ {bin_data} }};


void unload_{kernel_name}(void) {{
    HIP_CHECK(hipModuleUnload({kernel_name}_mod));
}}


void load_{kernel_name}() {{
    int dev = 0;
    void *bin = (void *)&HSACO_NAME;
    int shared = {shared};
    HIP_CHECK(hipModuleLoadData(&{kernel_name}_mod, bin));
    HIP_CHECK(hipModuleGetFunction(&{kernel_name}_func, {kernel_name}_mod, "{triton_kernel_name}"));
}}

/*
{kernel_docstring}
*/
hipError_t {kernel_name}(hipStream_t stream, {signature}) {{
    if ({kernel_name}_func == nullptr)
       load_{kernel_name}();
    unsigned int gX = {gridX};
    unsigned int gY = {gridY};
    unsigned int gZ = {gridZ};
    hipDeviceptr_t global_scratch = 0;
    hipDeviceptr_t profile_scratch = 0;
    void *args[{num_args}] = {{ {arg_pointers} }};
    // TODO: shared memory
    if(gX * gY * gZ > 0)
      return hipModuleLaunchKernel({kernel_name}_func, gX, gY, gZ, {num_warps} * warpSize, 1, 1, {shared}, stream, args, nullptr);
    else
      return hipErrorInvalidValue;
}}
