#ifndef TT_METAL_KERNEL_INCLUDES
#define TT_METAL_KERNEL_INCLUDES

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

/**
 * Result codes for Metal Triton kernel operations.
 */
typedef enum {
  MTL_TRITON_SUCCESS = 0,
  MTL_TRITON_ERROR_NO_DEVICE = -1,
  MTL_TRITON_ERROR_LIBRARY_LOAD = -2,
  MTL_TRITON_ERROR_FUNCTION_NOT_FOUND = -3,
  MTL_TRITON_ERROR_PIPELINE_CREATE = -4,
  MTL_TRITON_ERROR_COMMAND_QUEUE = -5,
  MTL_TRITON_ERROR_COMMAND_BUFFER = -6,
  MTL_TRITON_ERROR_ENCODER = -7,
  MTL_TRITON_ERROR_EXECUTION = -8,
  MTL_TRITON_ERROR_THREADGROUP_MEMORY = -9,
  MTL_TRITON_ERROR_INVALID_ARG = -10,
} MTLTritonResult;

/**
 * Opaque handle wrapping an MTLCommandQueue.
 * Pass NULL to let the runtime create its own queue.
 */
typedef void *MTLTritonQueue;

/**
 * Opaque handle wrapping an MTLDevice.
 */
typedef void *MTLTritonDevice;

/**
 * Opaque handle for a loaded Metal binary (metallib + pipeline state).
 */
typedef void *MTLTritonKernel;

/**
 * Load a .metallib binary and create a pipeline state for the named function.
 *
 * @param binary      Pointer to the metallib binary data.
 * @param binary_size Size of the metallib binary in bytes.
 * @param func_name   Name of the compute function within the metallib.
 * @param out_kernel  On success, receives a handle to the loaded kernel.
 * @return            MTL_TRITON_SUCCESS or an error code.
 */
MTLTritonResult mtl_load_binary(const void *binary, size_t binary_size,
                                const char *func_name,
                                MTLTritonKernel *out_kernel);

/**
 * Unload a previously loaded kernel and release associated resources.
 *
 * @param kernel  Handle returned by mtl_load_binary.
 */
void mtl_unload_binary(MTLTritonKernel kernel);

/**
 * Get the default Metal device handle.
 *
 * @param out_device  On success, receives an opaque device handle.
 * @return            MTL_TRITON_SUCCESS or MTL_TRITON_ERROR_NO_DEVICE.
 */
MTLTritonResult mtl_get_device(MTLTritonDevice *out_device);

/**
 * Launch a Metal compute kernel.
 *
 * @param kernel          Handle returned by mtl_load_binary.
 * @param queue           An MTLTritonQueue (bridged MTLCommandQueue), or NULL.
 * @param gridX           Number of threadgroups in X dimension.
 * @param gridY           Number of threadgroups in Y dimension.
 * @param gridZ           Number of threadgroups in Z dimension.
 * @param threads_per_tg  Threads per threadgroup (e.g. num_warps * warp_size).
 * @param shared_mem      Threadgroup memory size in bytes.
 * @param args            Array of argument pointers (buffers or value
 * pointers).
 * @param arg_sizes       Array of argument sizes in bytes.
 * @param num_args        Number of arguments.
 * @return                MTL_TRITON_SUCCESS or an error code.
 */
MTLTritonResult mtl_launch_kernel(MTLTritonKernel kernel, MTLTritonQueue queue,
                                  unsigned int gridX, unsigned int gridY,
                                  unsigned int gridZ,
                                  unsigned int threads_per_tg,
                                  unsigned int shared_mem, void **args,
                                  size_t *arg_sizes, int num_args);

#endif /* TT_METAL_KERNEL_INCLUDES */

// --- Template section (used by AOT code generator) ---
// tt-linker-backend: metal
//
// void unload_{kernel_name}(void);
// void load_{kernel_name}(void);
// tt-linker: {kernel_name}:{full_signature}:{algo_info}
// MTLTritonResult {kernel_name}(MTLTritonQueue queue, {signature});
