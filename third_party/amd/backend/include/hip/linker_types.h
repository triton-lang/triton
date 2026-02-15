
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

#ifndef HIP_INCLUDE_HIP_LINKER_TYPES_H
#define HIP_INCLUDE_HIP_LINKER_TYPES_H

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-identifier"
#pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#endif


#if !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#elif defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

/**
 *  @defgroup LinkerTypes Jit Linker Data Types
 *  @{
 *  This section describes the Jit Linker data types.
 *
 */

/**
 * hipJitOption
 */
typedef enum hipJitOption {
  hipJitOptionMaxRegisters = 0,         ///< CUDA Only Maximum registers may be used in a thread,
                                        ///< passed to compiler
  hipJitOptionThreadsPerBlock,          ///< CUDA Only Number of thread per block
  hipJitOptionWallTime,                 ///< CUDA Only Value for total wall clock time
  hipJitOptionInfoLogBuffer,            ///< CUDA Only Pointer to the buffer with logged information
  hipJitOptionInfoLogBufferSizeBytes,   ///< CUDA Only Size of the buffer in bytes for logged info
  hipJitOptionErrorLogBuffer,           ///< CUDA Only Pointer to the buffer with logged error(s)
  hipJitOptionErrorLogBufferSizeBytes,  ///< CUDA Only Size of the buffer in bytes for logged
                                        ///< error(s)
  hipJitOptionOptimizationLevel,  ///< Value of optimization level for generated codes, acceptable
                                  ///< options -O0, -O1, -O2, -O3
  hipJitOptionTargetFromContext,  ///< CUDA Only The target context, which is the default
  hipJitOptionTarget,             ///< CUDA Only JIT target
  hipJitOptionFallbackStrategy,   ///< CUDA Only Fallback strategy
  hipJitOptionGenerateDebugInfo,  ///< CUDA Only Generate debug information
  hipJitOptionLogVerbose,         ///< CUDA Only Generate log verbose
  hipJitOptionGenerateLineInfo,   ///< CUDA Only Generate line number information
  hipJitOptionCacheMode,          ///< CUDA Only Set cache mode
  hipJitOptionSm3xOpt,            ///< @deprecated CUDA Only New SM3X option.
  hipJitOptionFastCompile,        ///< CUDA Only Set fast compile
  hipJitOptionGlobalSymbolNames,  ///< CUDA Only Array of device symbol names to be relocated to the
                                  ///< host
  hipJitOptionGlobalSymbolAddresses,  ///< CUDA Only Array of host addresses to be relocated to the
                                      ///< device
  hipJitOptionGlobalSymbolCount,      ///< CUDA Only Number of symbol count.
  hipJitOptionLto,       ///< @deprecated CUDA Only Enable link-time optimization for device code
  hipJitOptionFtz,       ///< @deprecated CUDA Only Set single-precision denormals.
  hipJitOptionPrecDiv,   ///< @deprecated CUDA Only Set single-precision floating-point division
                         ///< and reciprocals
  hipJitOptionPrecSqrt,  ///< @deprecated CUDA Only Set single-precision floating-point square root
  hipJitOptionFma,       ///< @deprecated CUDA Only Enable floating-point multiplies and
                         ///< adds/subtracts operations
  hipJitOptionPositionIndependentCode,  ///< CUDA Only Generates Position Independent code
  hipJitOptionMinCTAPerSM,  ///< CUDA Only Hints to JIT compiler the minimum number of CTAs frin
                            ///< kernel's grid to be mapped to SM
  hipJitOptionMaxThreadsPerBlock,       ///< CUDA only Maximum number of threads in a thread block
  hipJitOptionOverrideDirectiveValues,  ///< Cuda only Override Directive values
  hipJitOptionNumOptions,               ///< Number of options
  hipJitOptionIRtoISAOptExt = 10000,    ///< Hip Only Linker options to be passed on to compiler
  hipJitOptionIRtoISAOptCountExt,  ///< Hip Only Count of linker options to be passed on to compiler
} hipJitOption;
/**
 * hipJitInputType
 */
typedef enum hipJitInputType {
  hipJitInputCubin = 0,                 ///< Cuda only Input cubin
  hipJitInputPtx,                       ///< Cuda only Input PTX
  hipJitInputFatBinary,                 ///< Cuda Only Input FAT Binary
  hipJitInputObject,                    ///< Cuda Only Host Object with embedded device code
  hipJitInputLibrary,                   ///< Cuda Only Archive of Host Objects with embedded
                                        ///< device code
  hipJitInputNvvm,                      ///< @deprecated Cuda only High Level intermediate
                                        ///< code for LTO
  hipJitNumLegacyInputTypes,            ///< Count of Legacy Input Types
  hipJitInputLLVMBitcode = 100,         ///< HIP Only LLVM Bitcode or IR assembly
  hipJitInputLLVMBundledBitcode = 101,  ///< HIP Only LLVM Clang Bundled Code
  hipJitInputLLVMArchivesOfBundledBitcode = 102,  ///< HIP Only LLVM Archive of Bundled Bitcode
  hipJitInputSpirv = 103,                         ///< HIP Only SPIRV Code Object
  hipJitNumInputTypes = 10                        ///< Count of Input Types
} hipJitInputType;
/**
 * hipJitCacheMode
 */
typedef enum hipJitCacheMode {
  hipJitCacheOptionNone = 0,
  hipJitCacheOptionCG,
  hipJitCacheOptionCA
} hipJitCacheMode;
/**
 * hipJitFallback
 */
typedef enum hipJitFallback {
  hipJitPreferPTX = 0,
  hipJitPreferBinary,
} hipJitFallback;

typedef enum hipLibraryOption_e {
  hipLibraryHostUniversalFunctionAndDataTable = 0,
  hipLibraryBinaryIsPreserved = 1
} hipLibraryOption;

// doxygen end LinkerTypes
/**
 * @}
 */

#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#endif  // HIP_INCLUDE_HIP_LINKER_TYPES_H