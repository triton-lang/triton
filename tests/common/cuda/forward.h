#ifndef _COMMON_CUDA_FORWARDS_H_
#define _COMMON_CUDA_FORwARDS_H_

struct cublasContext;
typedef struct cublasContext *cublasHandle_t;
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;

/* CUBLAS status type returns */
typedef enum{
    CUBLAS_STATUS_SUCCESS         =0,
    CUBLAS_STATUS_NOT_INITIALIZED =1,
    CUBLAS_STATUS_ALLOC_FAILED    =3,
    CUBLAS_STATUS_INVALID_VALUE   =7,
    CUBLAS_STATUS_ARCH_MISMATCH   =8,
    CUBLAS_STATUS_MAPPING_ERROR   =11,
    CUBLAS_STATUS_EXECUTION_FAILED=13,
    CUBLAS_STATUS_INTERNAL_ERROR  =14,
    CUBLAS_STATUS_NOT_SUPPORTED   =15,
    CUBLAS_STATUS_LICENSE_ERROR   =16
} cublasStatus_t;

/*For different GEMM algorithm */
typedef enum {
    CUBLAS_GEMM_DFALT               = -1,
    CUBLAS_GEMM_DEFAULT             = -1,
    CUBLAS_GEMM_ALGO0               =  0, // maxwell_sgemm_32x128_nt
    CUBLAS_GEMM_ALGO1               =  1, // maxwell_sgemm_64x64_nt
    CUBLAS_GEMM_ALGO2               =  2, // maxwell_sgemm_128x32_nt
    CUBLAS_GEMM_ALGO3               =  3, // maxwell_sgemm_128x64_nt
    CUBLAS_GEMM_ALGO4               =  4, // maxwell_sgemm_128x128_nt
    CUBLAS_GEMM_ALGO5               =  5,
    CUBLAS_GEMM_ALGO6               =  6,
    CUBLAS_GEMM_ALGO7               =  7,
    CUBLAS_GEMM_ALGO8               =  8,
    CUBLAS_GEMM_ALGO9               =  9,
    CUBLAS_GEMM_ALGO10              =  10,
    CUBLAS_GEMM_ALGO11              =  11,
    CUBLAS_GEMM_ALGO12              =  12,
    CUBLAS_GEMM_ALGO13              =  13,
    CUBLAS_GEMM_ALGO14              =  14,
    CUBLAS_GEMM_ALGO15              =  15,
    CUBLAS_GEMM_ALGO16              =  16,
    CUBLAS_GEMM_ALGO17              =  17,
    CUBLAS_GEMM_ALGO18              =  18, //sliced 32x32
    CUBLAS_GEMM_ALGO19              =  19, //sliced 64x32
    CUBLAS_GEMM_ALGO20              =  20, //sliced 128x32
    CUBLAS_GEMM_ALGO21              =  21, //sliced 32x32  -splitK
    CUBLAS_GEMM_ALGO22              =  22, //sliced 64x32  -splitK
    CUBLAS_GEMM_ALGO23              =  23, //sliced 128x32 -splitK
    CUBLAS_GEMM_DEFAULT_TENSOR_OP   =  99,
    CUBLAS_GEMM_DFALT_TENSOR_OP     =  99,
    CUBLAS_GEMM_ALGO0_TENSOR_OP     =  100,
    CUBLAS_GEMM_ALGO1_TENSOR_OP     =  101,
    CUBLAS_GEMM_ALGO2_TENSOR_OP     =  102,
    CUBLAS_GEMM_ALGO3_TENSOR_OP     =  103,
    CUBLAS_GEMM_ALGO4_TENSOR_OP     =  104,
    CUBLAS_GEMM_ALGO5_TENSOR_OP     =  105,
    CUBLAS_GEMM_ALGO6_TENSOR_OP     =  106,
    CUBLAS_GEMM_ALGO7_TENSOR_OP     =  107,
    CUBLAS_GEMM_ALGO8_TENSOR_OP     =  108,
    CUBLAS_GEMM_ALGO9_TENSOR_OP     =  109,
    CUBLAS_GEMM_ALGO10_TENSOR_OP     =  110,
    CUBLAS_GEMM_ALGO11_TENSOR_OP     =  111,
    CUBLAS_GEMM_ALGO12_TENSOR_OP     =  112,
    CUBLAS_GEMM_ALGO13_TENSOR_OP     =  113,
    CUBLAS_GEMM_ALGO14_TENSOR_OP     =  114,
    CUBLAS_GEMM_ALGO15_TENSOR_OP     =  115
} cublasGemmAlgo_t;

typedef enum cudaDataType_t
{
  CUDA_R_16F= 2,  /* real as a half */
  CUDA_C_16F= 6,  /* complex as a pair of half numbers */
  CUDA_R_32F= 0,  /* real as a float */
  CUDA_C_32F= 4,  /* complex as a pair of float numbers */
  CUDA_R_64F= 1,  /* real as a double */
  CUDA_C_64F= 5,  /* complex as a pair of double numbers */
  CUDA_R_8I = 3,  /* real as a signed char */
  CUDA_C_8I = 7,  /* complex as a pair of signed char numbers */
  CUDA_R_8U = 8,  /* real as a unsigned char */
  CUDA_C_8U = 9,  /* complex as a pair of unsigned char numbers */
  CUDA_R_32I= 10, /* real as a signed int */
  CUDA_C_32I= 11, /* complex as a pair of signed int numbers */
  CUDA_R_32U= 12, /* real as a unsigned int */
  CUDA_C_32U= 13  /* complex as a pair of unsigned int numbers */
} cudaDataType;

typedef cudaDataType cublasDataType_t;

typedef enum {
    CUBLAS_OP_N=0,
    CUBLAS_OP_T=1,
    CUBLAS_OP_C=2,
    CUBLAS_OP_HERMITAN=2, /* synonym if CUBLAS_OP_C */
    CUBLAS_OP_CONJG=3     /* conjugate */
} cublasOperation_t;

/*Enum for default math mode/tensor operation*/
typedef enum {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1
} cublasMath_t;

#endif
