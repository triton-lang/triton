#ifndef TRITON_HIPBLAS_TYPES_H
#define TRITON_HIPBLAS_TYPES_H

#include "hipblas-common.h"
#include <cstddef>
#include <cstdint>
// Forward declarations of hipBLAS types and functions.

typedef enum {
  HIPBLASLT_MATMUL_DESC_TRANSA =
      0, /**<Specifies the type of transformation operation that should be
            performed on matrix A. Default value is HIPBLAS_OP_N (for example,
            non-transpose operation). See hipblasOperation_t. Data
            Type:int32_t*/
  HIPBLASLT_MATMUL_DESC_TRANSB =
      1, /**<Specifies the type of transformation operation that should be
            performed on matrix B. Default value is HIPBLAS_OP_N (for example,
            non-transpose operation). See hipblasOperation_t. Data
            Type:int32_t*/
  HIPBLASLT_MATMUL_DESC_EPILOGUE =
      2, /**<Epilogue function. See hipblasLtEpilogue_t. Default value is:
            HIPBLASLT_EPILOGUE_DEFAULT. Data Type: uint32_t*/
  HIPBLASLT_MATMUL_DESC_BIAS_POINTER =
      3, /**<Bias or Bias gradient vector pointer in the device memory. Data
            Type:void* /const void* */
  HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE =
      4, /**<Type of the bias vector in the device memory. Can be set same as D
            matrix type or Scale type. Bias case: see HIPBLASLT_EPILOGUE_BIAS.
            Data Type:int32_t based on hipDataType*/
  HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER =
      5, /**<Device pointer to the scale factor value that converts data in
            matrix A to the compute data type range. The scaling factor must
            have the same type as the compute type. If not specified, or set to
            NULL, the scaling factor is assumed to be 1. If set for an
            unsupported matrix data, scale, and compute type combination,
            calling hipblasLtMatmul() will return HIPBLAS_INVALID_VALUE. Default
            value: NULL Data Type: void* /const void* */
  HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER =
      6, /**<Equivalent to HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER for matrix B.
            Default value: NULL Type: void* /const void* */
  HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER =
      7, /**<Equivalent to HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER for matrix C.
            Default value: NULL Type: void* /const void* */
  HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER =
      8, /**<Equivalent to HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER for matrix D.
            Default value: NULL Type: void* /const void* */
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER =
      9, /**<Equivalent to HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER for matrix AUX.
            Default value: NULL Type: void* /const void* */
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER =
      10, /**<Epilogue auxiliary buffer pointer in the device memory. Data
             Type:void* /const void* */
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD =
      11, /**<The leading dimension of the epilogue auxiliary buffer pointer in
             the device memory. Data Type:int64_t */
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE =
      12, /**<The batch stride of the epilogue auxiliary buffer pointer in the
             device memory. Data Type:int64_t */
  HIPBLASLT_MATMUL_DESC_POINTER_MODE =
      13, /**<Specifies alpha and beta are passed by reference, whether they are
             scalars on the host or on the device, or device vectors. Default
             value is: HIPBLASLT_POINTER_MODE_HOST (i.e., on the host). Data
             Type: int32_t based on hipblasLtPointerMode_t*/
  HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER =
      14, /**<Device pointer to the memory location that on completion will be
             set to the maximum of absolute values in the output matrix. Data
             Type:void* /const void* */
  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE =
      22, /**<Type of the aux vector in the device memory. Default value is:
             HIPBLASLT_DATATYPE_INVALID (using D matrix type). Data Type:int32_t
             based on hipDataType*/
  HIPBLASLT_MATMUL_DESC_A_SCALE_MODE =
      31, /**<Scaling mode that defines how the matrix scaling factor for matrix
             A is interpreted. See hipblasLtMatmulMatrixScale_t */
  HIPBLASLT_MATMUL_DESC_B_SCALE_MODE =
      32, /**<Scaling mode that defines how the matrix scaling factor for matrix
             B is interpreted. See hipblasLtMatmulMatrixScale_t */
  HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT =
      100, /**<Compute input A types. Defines the data type used for the input A
              of matrix multiply. */
  HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, /**<Compute input B types.
                                                     Defines the data type used
                                                     for the input B of matrix
                                                     multiply. */
  HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT, /**<first extra argument for the
                                                  activation function. Data
                                                  Type: float*/
  HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT, /**<second extra argument for the
                                                  activation function. Data
                                                  Type: float*/
  HIPBLASLT_MATMUL_DESC_MAX,
} hipblasLtMatmulDescAttributes_t;

typedef enum {
  HIPBLASLT_MATMUL_PREF_SEARCH_MODE = 0, /**<Search mode. Data Type: uint32_t*/
  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES =
      1, /**<Maximum allowed workspace memory. Default is 0 (no workspace memory
            allowed). Data Type: uint64_t*/
  HIPBLASLT_MATMUL_PREF_MAX = 2
} hipblasLtMatmulPreferenceAttributes_t;

typedef struct {
  uint64_t data[4];
} hipblasLtMatrixLayoutOpaque_t;
typedef hipblasLtMatrixLayoutOpaque_t *hipblasLtMatrixLayout_t;

typedef struct {
  uint64_t data[5];
} hipblasLtMatmulPreferenceOpaque_t;

typedef hipblasLtMatmulPreferenceOpaque_t *hipblasLtMatmulPreference_t;

typedef struct {
  uint64_t data[16] = {0};
  size_t max_workspace_bytes = 0;
} hipblasLtMatmulAlgo_t; // referencing all of this from rocm/rocm-libraries

typedef struct _hipblasLtMatmulHeuristicResult_t {
  hipblasLtMatmulAlgo_t algo; /**<Algo struct*/
  size_t workspaceSize = 0;   /**<Actual size of workspace memory required.*/
  hipblasStatus_t state =
      HIPBLAS_STATUS_SUCCESS; /**<Result status. Other fields are valid only if,
                                 after call to
                                 hipblasLtMatmulAlgoGetHeuristic(), this member
                                 is set to HIPBLAS_STATUS_SUCCESS..*/
  float wavesCount = 1.0;     /**<Waves count is a device utilization metric. A
                                 wavesCount value of 1.0f suggests that when the
                                 kernel is launched it will fully occupy the GPU.*/
  int reserved[4];            /**<Reserved.*/
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
