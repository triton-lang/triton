#ifndef TT_LINK_INCLUDES
#define TT_LINK_INCLUDES

#include <stdint.h>

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

typedef hipStream_t TT_StreamTy;
typedef hipError_t TT_ResultTy;

#define TT_ERROR_INVALID_VALUE hipErrorInvalidValue

#endif
