#ifndef TRITON_HIPBLAS_TYPES_H
#define TRITON_HIPBLAS_TYPES_H

#include "hipblas-common/hipblas-common.h"
#include <cstddef>
#include <cstdint>

// Forward declarations of hipBLAS types and functions.

typedef enum {
  HIPBLASLT_MATMUL_DESC_TRANSA = 0,
  HIPBLASLT_MATMUL_DESC_TRANSB = 1,
  HIPBLASLT_MATMUL_DESC_EPILOGUE = 2,
  HIPBLASLT_MATMUL_DESC_BIAS_POINTER = 3,
  HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 4,
  HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER = 5,
  HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER = 6,
  HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER = 7,
  HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER = 8,
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = 9,
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 10,
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 11,
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 12,
  HIPBLASLT_MATMUL_DESC_POINTER_MODE = 13,
  HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER = 14,
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = 22,
  HIPBLASLT_MATMUL_DESC_A_SCALE_MODE = 31,
  HIPBLASLT_MATMUL_DESC_B_SCALE_MODE = 32,
  HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT = 100,
  HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT,
  HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT,
  HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT,
  HIPBLASLT_MATMUL_DESC_MAX,
} hipblasLtMatmulDescAttributes_t;

typedef enum {
  HIPBLASLT_MATMUL_PREF_SEARCH_MODE = 0, /**<Search mode. Data Type: uint32_t*/
  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
  HIPBLASLT_MATMUL_PREF_MAX = 2
} hipblasLtMatmulPreferenceAttributes_t;

typedef struct hipblasLtMatrixLayoutOpaque_st {
  uint64_t data[4];
} hipblasLtMatrixLayoutOpaque_t;
typedef hipblasLtMatrixLayoutOpaque_t *hipblasLtMatrixLayout_t;

typedef struct hipblasLtMatmulPreferenceOpaque_st {
  uint64_t data[5];
} hipblasLtMatmulPreferenceOpaque_t;

typedef hipblasLtMatmulPreferenceOpaque_t *hipblasLtMatmulPreference_t;

typedef struct hipblasLtMatmulAlgo_st {
  uint64_t data[16];
  size_t max_workspace_bytes;
} hipblasLtMatmulAlgo_t; // referencing all of this from rocm/rocm-libraries

typedef struct _hipblasLtMatmulHeuristicResult_t {
  hipblasLtMatmulAlgo_t algo;
  size_t workspaceSize;
  hipblasStatus_t state;
  float wavesCount;
} hipblasLtMatmulHeuristicResult_t;

typedef enum hipDataType {
  HIP_R_32F = 0,
  HIP_R_64F = 1,
  HIP_R_16F = 2,
  HIP_R_8I = 3,
  HIP_C_32F = 4,
  HIP_C_64F = 5,
  HIP_C_16F = 6,
  HIP_C_8I = 7,
  HIP_R_8U = 8,
  HIP_C_8U = 9,
  HIP_R_32I = 10,
  HIP_C_32I = 11,
  HIP_R_32U = 12,
  HIP_C_32U = 13,
  HIP_R_16BF = 14,
  HIP_C_16BF = 15,
  HIP_R_4I = 16,
  HIP_C_4I = 17,
  HIP_R_4U = 18,
  HIP_C_4U = 19,
  HIP_R_16I = 20,
  HIP_C_16I = 21,
  HIP_R_16U = 22,
  HIP_C_16U = 23,
  HIP_R_64I = 24,
  HIP_C_64I = 25,
  HIP_R_64U = 26,
  HIP_C_64U = 27,
  HIP_R_8F_E4M3 = 28,
  HIP_R_8F_E5M2 = 29,
  HIP_R_8F_UE8M0 = 30,
  HIP_R_6F_E2M3 = 31,
  HIP_R_6F_E3M2 = 32,
  HIP_R_4F_E2M1 = 33,
  // HIP specific Data Types
  HIP_R_8F_E4M3_FNUZ = 1000,
  HIP_R_8F_E5M2_FNUZ = 1001,
} hipDataType;

struct hipblasContext;
typedef struct hipblasLtContext *hipblasLtHandle_t;
struct hipblasLtMatmulDescOpaque_t;
typedef hipblasLtMatmulDescOpaque_t *hipblasLtMatmulDesc_t;
struct hipStream_st;
typedef struct hipStream_st *hipStream_t;

#endif // TRITON_HIPBLAS_TYPES_H
