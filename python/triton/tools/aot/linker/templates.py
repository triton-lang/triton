"""Header Templates"""
DEFAULT_ALGO_DECL_TEMPLATE = """
CUresult {name}(CUstream stream, {args});
void load_{name}();
void unload_{name}();
"""

DEFAULT_GLOBAL_DECL_TEMPLATE = """
CUresult {orig_kernel_name}_default(CUstream stream, {default_args});
CUresult {orig_kernel_name}(CUstream stream, {full_args}, int algo_id);
void load_{orig_kernel_name}();
void unload_{orig_kernel_name}();
"""

DEFAULT_HEADER_INCLUDES = ["#include <cuda.h>"]
"""Source Templates"""
DEFAULT_SOURCE_INCLUDES = [
    "#include <cuda.h>",
    "#include <stdint.h>",
    "#include <assert.h>",
]
