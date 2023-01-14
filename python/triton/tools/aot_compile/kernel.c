/*[ kernel_header ]*/
#include "common.h"

unsigned char {binary_arg_name}[{bin_size}];
void load_{kernel_name}(void);
CUresult {kernel_name}(CUstream stream, GridWarps g, {signature});
CUmodule {kernel_name}_mod;
CUfunction {kernel_name}_func;

/*[ default_load ]*/
CUmodule {kernel_name}_mod = NULL;
CUfunction {kernel_name}_func = NULL;

unsigned char {binary_arg_name}[{bin_size}] = {{ {binary_val} }};

// TODO: make unloader function(?)
void load_{kernel_name}(void)
/*
    This loads the module and kernel to the global {kernel_name}_mod and {kernel_name}_func 
*/
{{
    CUresult err;
    void *image = (void *)&{binary_arg_name};
    err = cuModuleLoadData(&{kernel_name}_mod, image);
    if (err != CUDA_SUCCESS) {{
        // TODO: tell the user where things went wrong 
        printf("Error Module Load: %d\n", err);
        return;
    }}
    err = cuModuleGetFunction(&{kernel_name}_func, {kernel_name}_mod, "{compiled_func_name}");
    if (err != CUDA_SUCCESS) {{
        // TODO: tell the user where things went wrong 
        printf("Error Function Load: %d\n", err);
        return;
    }}
    return;
}}

/*[ default_launch ]*/
CUresult {kernel_name}(CUstream stream, GridWarps g, {signature})
/*
    {kernel_docstring}
*/

{{
    if ({kernel_name}_func == NULL) {{
       load_{kernel_name}(); 
    }}
    void *args[{num_args}] = {{ {arg_pointers} }};
    return cuLaunchKernel({kernel_name}_func, g.gX, g.gY, g.gZ, g.numWarps * {threads_per_warp}, 1, 1, 0, stream, args, NULL);
}}

/*[ user_launch ]*/
CUresult launch_{kernel_name}(CUstream stream, GridWarps g, CUfunction* func, {signature}) 
/*
    This is a launch function, to let the user handle its own loading.
    To use the default kernel loding (cubin stored in source) use {kernel_name}

    {kernel_docstring}
*/
{{
    void *args[{num_args}] = {{ {arg_pointers} }};
    return cuLaunchKernel(*func, g.gX, g.gY, g.gZ, g.numWarps * {threads_per_warp}, 1, 1, 0, stream, args, NULL);
}}
