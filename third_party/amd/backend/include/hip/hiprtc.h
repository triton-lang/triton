/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once

#include <hip/hip_common.h>

#if !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/nvidia_detail/nvidia_hiprtc.h>
#elif defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdlib.h>

#if !defined(_WIN32)
#pragma GCC visibility push(default)
#endif

/**
 *
 * @addtogroup GlobalDefs
 * @{
 *  
 */
 /**
 * hiprtc error code
 */
typedef enum hiprtcResult {
    HIPRTC_SUCCESS = 0,   ///< Success
    HIPRTC_ERROR_OUT_OF_MEMORY = 1,  ///< Out of memory
    HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,   ///< Failed to create program
    HIPRTC_ERROR_INVALID_INPUT = 3,   ///< Invalid input
    HIPRTC_ERROR_INVALID_PROGRAM = 4,   ///< Invalid program
    HIPRTC_ERROR_INVALID_OPTION = 5,   ///< Invalid option
    HIPRTC_ERROR_COMPILATION = 6,   ///< Compilation error
    HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,   ///< Failed in builtin operation
    HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,   ///< No name expression after compilation
    HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,   ///< No lowered names before compilation 
    HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,   ///< Invalid name expression
    HIPRTC_ERROR_INTERNAL_ERROR = 11,   ///< Internal error
    HIPRTC_ERROR_LINKING = 100   ///< Error in linking
} hiprtcResult;

/**
 * hiprtc JIT option
 */

typedef enum hiprtcJIT_option {
  HIPRTC_JIT_MAX_REGISTERS = 0,  ///< CUDA Only Maximum registers may be used in a thread, passed to compiler
  HIPRTC_JIT_THREADS_PER_BLOCK,  ///< CUDA Only Number of thread per block
  HIPRTC_JIT_WALL_TIME,  ///< CUDA Only Value for total wall clock time
  HIPRTC_JIT_INFO_LOG_BUFFER,  ///< CUDA Only Pointer to the buffer with logged information
  HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES,  ///< CUDA Only Size of the buffer in bytes for logged info
  HIPRTC_JIT_ERROR_LOG_BUFFER,  ///< CUDA Only Pointer to the buffer with logged error(s)
  HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,  ///< CUDA Only Size of the buffer in bytes for logged error(s)
  HIPRTC_JIT_OPTIMIZATION_LEVEL,  ///< Value of optimization level for generated codes, acceptable options -O0, -O1, -O2, -O3
  HIPRTC_JIT_TARGET_FROM_HIPCONTEXT,  ///< CUDA Only The target context, which is the default
  HIPRTC_JIT_TARGET,  ///< CUDA Only JIT target
  HIPRTC_JIT_FALLBACK_STRATEGY,  ///< CUDA Only Fallback strategy
  HIPRTC_JIT_GENERATE_DEBUG_INFO,  ///< CUDA Only Generate debug information
  HIPRTC_JIT_LOG_VERBOSE,  ///< CUDA Only Generate log verbose
  HIPRTC_JIT_GENERATE_LINE_INFO,  ///< CUDA Only Generate line number information
  HIPRTC_JIT_CACHE_MODE,  ///< CUDA Only Set cache mode
  HIPRTC_JIT_NEW_SM3X_OPT,  ///< @deprecated CUDA Only New SM3X option.
  HIPRTC_JIT_FAST_COMPILE,  ///< CUDA Only Set fast compile
  HIPRTC_JIT_GLOBAL_SYMBOL_NAMES,  ///< CUDA Only Array of device symbol names to be relocated to the host
  HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS,  ///< CUDA Only Array of host addresses to be relocated to the device
  HIPRTC_JIT_GLOBAL_SYMBOL_COUNT,  ///< CUDA Only Number of symbol count.
  HIPRTC_JIT_LTO,  ///< @deprecated CUDA Only Enable link-time optimization for device code
  HIPRTC_JIT_FTZ,  ///< @deprecated CUDA Only Set single-precision denormals.
  HIPRTC_JIT_PREC_DIV,  ///< @deprecated CUDA Only Set single-precision floating-point division and
                        ///< reciprocals
  HIPRTC_JIT_PREC_SQRT,  ///< @deprecated CUDA Only Set single-precision floating-point square root
  HIPRTC_JIT_FMA,  ///< @deprecated CUDA Only Enable floating-point multiplies and adds/subtracts operations
  HIPRTC_JIT_NUM_OPTIONS,  ///< Number of options
  HIPRTC_JIT_IR_TO_ISA_OPT_EXT = 10000,  ///< Linker options to be passed on to compiler
                                         /// @note  Only supported for the AMD platform.
  HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT,    ///< Count of linker options to be passed on to
                                         ///< compiler  @note  Only supported for the AMD platform
} hiprtcJIT_option;

/**
 * hiprtc JIT input type
 */
typedef enum hiprtcJITInputType {
  HIPRTC_JIT_INPUT_CUBIN = 0,  ///< Input cubin
  HIPRTC_JIT_INPUT_PTX,  ///< Input PTX
  HIPRTC_JIT_INPUT_FATBINARY,  ///< Input fat binary
  HIPRTC_JIT_INPUT_OBJECT,  ///< Input object
  HIPRTC_JIT_INPUT_LIBRARY,  ///< Input library
  HIPRTC_JIT_INPUT_NVVM,  ///< Input NVVM
  HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES,  ///< Number of legacy input type
  HIPRTC_JIT_INPUT_LLVM_BITCODE = 100,  ///< LLVM bitcode or IR assembly
  HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE = 101,  ///< LLVM bundled bitcode
  HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE = 102,  ///< LLVM archives of boundled bitcode
  HIPRTC_JIT_NUM_INPUT_TYPES = (HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES + 3)
} hiprtcJITInputType;
/**
* @}
*/

/**
 *  hiprtc link state
 *
 */
typedef struct ihiprtcLinkState* hiprtcLinkState;
/**
 *  @ingroup Runtime
 *
 * @brief Returns text string message to explain the error which occurred
 *
 * @param [in] result  code to convert to string.
 * @returns  const char pointer to the NULL-terminated error string
 *
 * @warning In HIP, this function returns the name of the error,
 * if the hiprtc result is defined, it will return "Invalid HIPRTC error code"
 *
 * @see hiprtcResult
 */
const char* hiprtcGetErrorString(hiprtcResult result);

/**
 * @ingroup Runtime
 * @brief Sets the parameters as major and minor version.
 *
 * @param [out] major  HIP Runtime Compilation major version.
 * @param [out] minor  HIP Runtime Compilation minor version.
 *
 * @returns #HIPRTC_ERROR_INVALID_INPUT, #HIPRTC_SUCCESS
 *
 */
hiprtcResult hiprtcVersion(int* major, int* minor);

/**
 *  hiprtc program
 *
 */
typedef struct _hiprtcProgram* hiprtcProgram;

/**
 * @ingroup Runtime
 * @brief Adds the given name exprssion to the runtime compilation program.
 *
 * @param [in] prog  runtime compilation program instance.
 * @param [in] name_expression  const char pointer to the name expression.
 * @returns  #HIPRTC_SUCCESS
 *
 * If const char pointer is NULL, it will return #HIPRTC_ERROR_INVALID_INPUT.
 *
 * @see hiprtcResult
 */
hiprtcResult hiprtcAddNameExpression(hiprtcProgram prog,
                                     const char* name_expression);

/**
 * @ingroup Runtime
 * @brief Compiles the given runtime compilation program.
 *
 * @param [in] prog  runtime compilation program instance.
 * @param [in] numOptions  number of compiler options.
 * @param [in] options  compiler options as const array of strins.
 * @returns #HIPRTC_SUCCESS
 *
 * If the compiler failed to build the runtime compilation program,
 * it will return #HIPRTC_ERROR_COMPILATION.
 *
 * @see hiprtcResult
 */
hiprtcResult hiprtcCompileProgram(hiprtcProgram prog,
                                  int numOptions,
                                  const char** options);

/**
 * @ingroup Runtime
 * @brief Creates an instance of hiprtcProgram with the given input parameters,
 * and sets the output hiprtcProgram prog with it.
 *
 * @param [in, out] prog  runtime compilation program instance.
 * @param [in] src  const char pointer to the program source.
 * @param [in] name  const char pointer to the program name.
 * @param [in] numHeaders  number of headers.
 * @param [in] headers  array of strings pointing to headers.
 * @param [in] includeNames  array of strings pointing to names included in program source.
 * @returns #HIPRTC_SUCCESS
 *
 * Any invalide input parameter, it will return #HIPRTC_ERROR_INVALID_INPUT
 * or #HIPRTC_ERROR_INVALID_PROGRAM.
 *
 * If failed to create the program, it will return #HIPRTC_ERROR_PROGRAM_CREATION_FAILURE.
 *
 * @see hiprtcResult
 */
hiprtcResult hiprtcCreateProgram(hiprtcProgram* prog,
                                 const char* src,
                                 const char* name,
                                 int numHeaders,
                                 const char** headers,
                                 const char** includeNames);

/**
 * @brief Destroys an instance of given hiprtcProgram.
 * @ingroup Runtime
 * @param [in] prog  runtime compilation program instance.
 * @returns #HIPRTC_SUCCESS
 *
 * If prog is NULL, it will return #HIPRTC_ERROR_INVALID_INPUT.
 *
 * @see hiprtcResult
 */
hiprtcResult hiprtcDestroyProgram(hiprtcProgram* prog);

/**
 * @brief Gets the lowered (mangled) name from an instance of hiprtcProgram with the given input parameters,
 * and sets the output lowered_name with it.
 * @ingroup Runtime
 * @param [in] prog  runtime compilation program instance.
 * @param [in] name_expression  const char pointer to the name expression.
 * @param [in, out] lowered_name  const char array to the lowered (mangled) name.
 * @returns #HIPRTC_SUCCESS
 *
 * If any invalide nullptr input parameters, it will return #HIPRTC_ERROR_INVALID_INPUT
 *
 * If name_expression is not found, it will return #HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
 *
 * If failed to get lowered_name from the program, it will return #HIPRTC_ERROR_COMPILATION.
 *
 * @see hiprtcResult
 */
hiprtcResult hiprtcGetLoweredName(hiprtcProgram prog,
                                  const char* name_expression,
                                  const char** lowered_name);

/**
 * @brief Gets the log generated by the runtime compilation program instance.
 * @ingroup Runtime
 * @param [in] prog  runtime compilation program instance.
 * @param [out] log  memory pointer to the generated log.
 * @returns #HIPRTC_SUCCESS
 *
 * @see hiprtcResult
 */
hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char* log);

/**
 * @brief Gets the size of log generated by the runtime compilation program instance.
 *
 * @param [in] prog  runtime compilation program instance.
 * @param [out] logSizeRet  size of generated log.
 * @returns #HIPRTC_SUCCESS
 *
 * @see hiprtcResult
 */
hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog,
                                     size_t* logSizeRet);

/**
 * @brief Gets the pointer of compilation binary by the runtime compilation program instance.
 * @ingroup Runtime
 * @param [in] prog  runtime compilation program instance.
 * @param [out] code  char pointer to binary.
 * @returns #HIPRTC_SUCCESS
 *
 * @see hiprtcResult
 */
hiprtcResult hiprtcGetCode(hiprtcProgram prog, char* code);

/**
 * @brief Gets the size of compilation binary by the runtime compilation program instance.
 * @ingroup Runtime
 * @param [in] prog  runtime compilation program instance.
 * @param [out] codeSizeRet  the size of binary.
 * @returns #HIPRTC_SUCCESS
 *
 * @see hiprtcResult
 */
hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t* codeSizeRet);

/**
 * @brief Gets the pointer of compiled bitcode by the runtime compilation program instance.
 *
 * @param [in] prog  runtime compilation program instance.
 * @param [out] bitcode  char pointer to bitcode.
 * @return HIPRTC_SUCCESS
 *
 * @see hiprtcResult
 */
hiprtcResult hiprtcGetBitcode(hiprtcProgram prog, char* bitcode);

/**
 * @brief Gets the size of compiled bitcode by the runtime compilation program instance.
 * @ingroup Runtime
 *
 * @param [in] prog  runtime compilation program instance.
 * @param [out] bitcode_size  the size of bitcode.
 * @returns #HIPRTC_SUCCESS
 *
 * @see hiprtcResult
 */
hiprtcResult hiprtcGetBitcodeSize(hiprtcProgram prog, size_t* bitcode_size);

/**
 * @brief Creates the link instance via hiprtc APIs.
 * @ingroup Runtime
 * @param [in] num_options  Number of options
 * @param [in] option_ptr  Array of options
 * @param [in] option_vals_pptr  Array of option values cast to void*
 * @param [out] hip_link_state_ptr  hiprtc link state created upon success
 *
 * @returns #HIPRTC_SUCCESS, #HIPRTC_ERROR_INVALID_INPUT, #HIPRTC_ERROR_INVALID_OPTION
 *
 * @see hiprtcResult
 */
hiprtcResult hiprtcLinkCreate(unsigned int num_options, hiprtcJIT_option* option_ptr,
                              void** option_vals_pptr, hiprtcLinkState* hip_link_state_ptr);

/**
 * @brief Adds a file with bit code to be linked with options
 * @ingroup Runtime
 * @param [in] hip_link_state  hiprtc link state
 * @param [in] input_type  Type of the input data or bitcode
 * @param [in] file_path  Path to the input file where bitcode is present
 * @param [in] num_options  Size of the options
 * @param [in] options_ptr  Array of options applied to this input
 * @param [in] option_values  Array of option values cast to void*
 *
 * @returns #HIPRTC_SUCCESS
 *
 * If input values are invalid, it will
 * @return #HIPRTC_ERROR_INVALID_INPUT
 *
 * @see hiprtcResult
 */

hiprtcResult hiprtcLinkAddFile(hiprtcLinkState hip_link_state, hiprtcJITInputType input_type,
                               const char* file_path, unsigned int num_options,
                               hiprtcJIT_option* options_ptr, void** option_values);

/**
 * @brief Completes the linking of the given program.
 * @ingroup Runtime
 * @param [in] hip_link_state  hiprtc link state
 * @param [in] input_type  Type of the input data or bitcode
 * @param [in] image  Input data which is null terminated
 * @param [in] image_size  Size of the input data
 * @param [in] name  Optional name for this input
 * @param [in] num_options  Size of the options
 * @param [in] options_ptr  Array of options applied to this input
 * @param [in] option_values  Array of option values cast to void*
 *
 * @returns #HIPRTC_SUCCESS, #HIPRTC_ERROR_INVALID_INPUT
 *
 * If adding the file fails, it will
 * @return #HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
 *
 * @see hiprtcResult
 */

hiprtcResult hiprtcLinkAddData(hiprtcLinkState hip_link_state, hiprtcJITInputType input_type,
                               void* image, size_t image_size, const char* name,
                               unsigned int num_options, hiprtcJIT_option* options_ptr,
                               void** option_values);

/**
 * @brief Completes the linking of the given program.
 * @ingroup Runtime
 * @param [in]  hip_link_state  hiprtc link state
 * @param [out]  bin_out  Upon success, points to the output binary
 * @param [out]  size_out  Size of the binary is stored (optional)
 *
 * @returns #HIPRTC_SUCCESS
 *
 * If adding the data fails, it will
 * @return #HIPRTC_ERROR_LINKING
 *
 * @see hiprtcResult
 */
hiprtcResult hiprtcLinkComplete(hiprtcLinkState hip_link_state, void** bin_out, size_t* size_out);

/**
 * @brief Deletes the link instance via hiprtc APIs.
 * @ingroup Runtime
 * @param [in] hip_link_state link state instance
 *
 * @returns #HIPRTC_SUCCESS
 *
 * @see hiprtcResult
 */
hiprtcResult hiprtcLinkDestroy(hiprtcLinkState hip_link_state);

#if !defined(_WIN32)
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
