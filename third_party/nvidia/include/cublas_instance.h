#ifndef TRITON_CUBLAS_INSTANCE_H
#define TRITON_CUBLAS_INSTANCE_H

#include "cublas_types.h"
#include <dlfcn.h>
#include <stdexcept>
#include <string>

class CublasLtInstance {
private:
  // Typedefs for cublas functions
  typedef cublasStatus_t (*cublasLtCreate_t)(cublasLtHandle_t *);
  typedef cublasStatus_t (*cublasLtDestroy_t)(cublasLtHandle_t);
  typedef cublasStatus_t (*cublasLtMatmulDescCreate_t)(cublasLtMatmulDesc_t *,
                                                       cublasComputeType_t,
                                                       cudaDataType_t);
  typedef cublasStatus_t (*cublasLtMatmulDescDestroy_t)(cublasLtMatmulDesc_t);
  typedef cublasStatus_t (*cublasLtMatmulDescSetAttribute_t)(
      cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void *,
      size_t);
  typedef cublasStatus_t (*cublasLtMatrixLayoutCreate_t)(
      cublasLtMatrixLayout_t *, cudaDataType_t, uint64_t, uint64_t, int64_t);
  typedef cublasStatus_t (*cublasLtMatrixLayoutDestroy_t)(
      cublasLtMatrixLayout_t);
  typedef cublasStatus_t (*cublasLtMatmulPreferenceCreate_t)(
      cublasLtMatmulPreference_t *);
  typedef cublasStatus_t (*cublasLtMatmulPreferenceDestroy_t)(
      cublasLtMatmulPreference_t);
  typedef cublasStatus_t (*cublasLtMatmulPreferenceSetAttribute_t)(
      cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t,
      const void *, size_t);
  typedef cublasStatus_t (*cublasLtMatmulAlgoGetHeuristic_t)(
      cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
      cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
      cublasLtMatmulPreference_t, int, cublasLtMatmulHeuristicResult_t *,
      int *);
  typedef cublasStatus_t (*cublasLtMatmul_t)(
      cublasLtHandle_t, cublasLtMatmulDesc_t, const void *, const void *,
      const cublasLtMatrixLayout_t, const void *, const cublasLtMatrixLayout_t,
      const void *, const void *, const cublasLtMatrixLayout_t, void *,
      const cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t *, void *,
      size_t, cudaStream_t);

  static constexpr const char *name = "libcublas.so";

  cublasLtCreate_t cublasLtCreate;
  cublasLtDestroy_t cublasLtDestroy;
  cublasLtMatmulDescCreate_t cublasLtMatmulDescCreate;
  cublasLtMatmulDescDestroy_t cublasLtMatmulDescDestroy;
  cublasLtMatmulDescSetAttribute_t cublasLtMatmulDescSetAttribute;
  cublasLtMatrixLayoutCreate_t cublasLtMatrixLayoutCreate;
  cublasLtMatrixLayoutDestroy_t cublasLtMatrixLayoutDestroy;
  cublasLtMatmulPreferenceCreate_t cublasLtMatmulPreferenceCreate;
  cublasLtMatmulPreferenceDestroy_t cublasLtMatmulPreferenceDestroy;
  cublasLtMatmulPreferenceSetAttribute_t cublasLtMatmulPreferenceSetAttribute;
  cublasLtMatmulAlgoGetHeuristic_t cublasLtMatmulAlgoGetHeuristic;
  cublasLtMatmul_t cublasLtMatmul;

  void *dylibHandle = nullptr;
  cublasLtHandle_t ltHandle;

  void *workspace = nullptr;
  size_t workspaceSize = 0;

  cublasLtMatmulPreference_t preference = NULL;

  void loadCublasDylib() {
    if (dylibHandle == nullptr) {
      // First reuse the existing handle
      dylibHandle = dlopen(name, RTLD_NOLOAD);
    }
    if (dylibHandle == nullptr) {
      // If not found, try to load it
      dylibHandle = dlopen(name, RTLD_LOCAL | RTLD_LAZY);
    }
    if (dylibHandle == nullptr) {
      throw std::runtime_error("Could not find `" + std::string(name) +
                               "`. Make sure it is in your "
                               "LD_LIBRARY_PATH.");
    }
    dlerror(); // Clear any existing error

    cublasLtCreate = (cublasLtCreate_t)dlsym(dylibHandle, "cublasLtCreate");
    cublasLtDestroy = (cublasLtDestroy_t)dlsym(dylibHandle, "cublasLtDestroy");
    cublasLtMatmulDescCreate = (cublasLtMatmulDescCreate_t)dlsym(
        dylibHandle, "cublasLtMatmulDescCreate");
    cublasLtMatmulDescDestroy = (cublasLtMatmulDescDestroy_t)dlsym(
        dylibHandle, "cublasLtMatmulDescDestroy");
    cublasLtMatmulDescSetAttribute = (cublasLtMatmulDescSetAttribute_t)dlsym(
        dylibHandle, "cublasLtMatmulDescSetAttribute");
    cublasLtMatrixLayoutCreate = (cublasLtMatrixLayoutCreate_t)dlsym(
        dylibHandle, "cublasLtMatrixLayoutCreate");
    cublasLtMatrixLayoutDestroy = (cublasLtMatrixLayoutDestroy_t)dlsym(
        dylibHandle, "cublasLtMatrixLayoutDestroy");
    cublasLtMatmulPreferenceCreate = (cublasLtMatmulPreferenceCreate_t)dlsym(
        dylibHandle, "cublasLtMatmulPreferenceCreate");
    cublasLtMatmulPreferenceDestroy = (cublasLtMatmulPreferenceDestroy_t)dlsym(
        dylibHandle, "cublasLtMatmulPreferenceDestroy");
    cublasLtMatmulPreferenceSetAttribute =
        (cublasLtMatmulPreferenceSetAttribute_t)dlsym(
            dylibHandle, "cublasLtMatmulPreferenceSetAttribute");
    cublasLtMatmulAlgoGetHeuristic = (cublasLtMatmulAlgoGetHeuristic_t)dlsym(
        dylibHandle, "cublasLtMatmulAlgoGetHeuristic");
    cublasLtMatmul = (cublasLtMatmul_t)dlsym(dylibHandle, "cublasLtMatmul");

    const char *dlsym_error = dlerror();
    if (dlsym_error) {
      throw std::runtime_error("Could not load symbol from `" +
                               std::string(name) +
                               "`: " + std::string(dlsym_error));
    }
  }

  void unloadCublasDylib() {
    if (dylibHandle)
      dlclose(dylibHandle);
  }

  void successOrExit(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cuBLAS Error: " + std::to_string(status) +
                               "\n");
    }
  }

  // Simple wrapper around the cublasLtMatmul function
  void gemm_impl(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
                 uint64_t D, cudaDataType_t dtype, float alpha, float beta) {
    cublasLtMatmulDesc_t matmulDesc = NULL;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    int8_t fastAccum = 1;

    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL,
                           Ddesc = NULL;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // Select compute type. Use TF32 when inputs are FP32, otherwise default
    // FP32 accumulation.
    cublasComputeType_t computeType = (dtype == CUDA_R_32F)
                                          ? CUBLAS_COMPUTE_32F_FAST_TF32
                                          : CUBLAS_COMPUTE_32F;
    successOrExit(
        cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32F));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    if (dtype == CUDA_R_8F_E4M3) {
      successOrExit(cublasLtMatmulDescSetAttribute(
          matmulDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum,
          sizeof(fastAccum)));
    }

    auto c_dtype = dtype == CUDA_R_8F_E4M3 ? CUDA_R_16F : dtype;
    successOrExit(cublasLtMatrixLayoutCreate(&Adesc, dtype, k, m, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Bdesc, dtype, k, n, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Cdesc, c_dtype, m, n, m));
    successOrExit(cublasLtMatrixLayoutCreate(&Ddesc, dtype, m, n, m));

    successOrExit(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
        &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
      throw std::runtime_error(
          "No valid algorithm found by cublasLtMatmulAlgoGetHeuristic");
    }

    successOrExit(cublasLtMatmul(ltHandle, matmulDesc, &alpha, (void *)A, Adesc,
                                 (void *)B, Bdesc, &beta, (void *)C, Cdesc,
                                 (void *)D, Ddesc, &heuristicResult.algo,
                                 (void *)workspace, workspaceSize, 0));
    if (Ddesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc)
      successOrExit(cublasLtMatmulDescDestroy(matmulDesc));
  }

  // Block-scaled matmul: D = (A * scale_A) @ (B * scale_B)
  //
  // Supports two modes via is_mxfp8 parameter:
  //   - MXFP8 (is_mxfp8=true):  FP8 E4M3 inputs, E8M0 scales (32-element
  //   groups)
  //   - NVFP4 (is_mxfp8=false): FP4 E2M1 inputs, FP8 E4M3 scales (16-element
  //   groups)
  //
  // Input layout requirements (row-major):
  //   - A: (M, K) in FP8/FP4 (FP4 is packed, 2 elements per byte)
  //   - B: (N, K) in FP8/FP4 (caller must transpose B before calling)
  //   - scale_A, scale_B: scale factors for block scaling
  //   - Output D: (M, N) in FP16
  //
  // Note: cuBLAS uses column-major layout. This function internally swaps
  // A and B operands and applies transposes to handle the conversion.
  void block_scaled_matmul(int m, int n, int k, uint64_t A, uint64_t B,
                           uint64_t D_out, uint64_t scale_A, uint64_t scale_B,
                           bool is_mxfp8) {
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL,
                           Ddesc = NULL;

    // Use FP32 compute and accumulation
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    successOrExit(
        cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32F));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Enable fast accumulation for MXFP8 only
    // "Flag for managing FP8 fast accumulation mode. When enabled, on some GPUs
    //  problem execution might be faster but at the cost of lower accuracy
    //  because intermediate results will not periodically be promoted to a
    //  higher precision. Currently this flag has an effect on the following
    //  GPUs: Ada, Hopper.""
    if (is_mxfp8) {
      int8_t fastAccum = 1;
      successOrExit(cublasLtMatmulDescSetAttribute(
          matmulDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum,
          sizeof(fastAccum)));
    }

    // Set scale mode based on format
    // MXFP8: 32-element groups with E8M0 scales
    // NVFP4: 16-element groups with FP8 E4M3 scales
    cublasLtMatmulMatrixScale_t ab_scale_type =
        is_mxfp8 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
                 : CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;

    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &ab_scale_type,
        sizeof(ab_scale_type)));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &ab_scale_type,
        sizeof(ab_scale_type)));

    // Set scale POINTERS
    // NOTE: A and B matrices are swapped in cublasLtMatmul call to handle
    // row-major vs column-major conversion.
    void *scale_A_ptr = (void *)scale_A;
    void *scale_B_ptr = (void *)scale_B;
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_B_ptr,
        sizeof(scale_B_ptr))); // Swapped
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_A_ptr,
        sizeof(scale_A_ptr))); // Swapped

    // Create matrix layouts
    // MXFP8: CUDA_R_8F_E4M3, NVFP4: CUDA_R_4F_E2M1
    // With transa=T: A layout is (k, m), lda=k
    // With transb=N: B layout is (k, n), ldb=k
    cudaDataType_t dataType = is_mxfp8 ? CUDA_R_8F_E4M3 : CUDA_R_4F_E2M1;
    successOrExit(cublasLtMatrixLayoutCreate(&Adesc, dataType, k, m, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Bdesc, dataType, k, n, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, m));
    Ddesc = Cdesc;

    float alpha = 1.0f;
    float beta = 0.0f; // No bias

    // Query cuBLAS heuristics for the best algorithm
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1,
        &heuristicResult, &returnedResults);

    if (status != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
      throw std::runtime_error(
          "cublasLtMatmulAlgoGetHeuristic failed (status=" +
          std::to_string(status) +
          ", results=" + std::to_string(returnedResults) + ") for " +
          (is_mxfp8 ? "mxfp8" : "nvfp4"));
    }

    // Execute matmul with the selected algorithm
    // B and A are swapped for row-major to col-major conversion
    successOrExit(cublasLtMatmul(ltHandle, matmulDesc, &alpha, (void *)B, Bdesc,
                                 (void *)A, Adesc, &beta, (void *)D_out, Cdesc,
                                 (void *)D_out, Cdesc, &heuristicResult.algo,
                                 workspace, workspaceSize, 0));

    // Cleanup
    if (Cdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc)
      successOrExit(cublasLtMatmulDescDestroy(matmulDesc));
  }

public:
  CublasLtInstance(uint64_t workspace, size_t workspaceSize)
      : workspace((void *)workspace), workspaceSize(workspaceSize) {
    loadCublasDylib();
    cublasLtCreate(&ltHandle);

    successOrExit(cublasLtMatmulPreferenceCreate(&preference));
    successOrExit(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));
  }
  ~CublasLtInstance() {
    if (preference)
      successOrExit(cublasLtMatmulPreferenceDestroy(preference));

    cublasLtDestroy(ltHandle);
    unloadCublasDylib();
  }

  // C = A * B
  // Matrix B needs to be transposed, while matrix A does not. The function
  // *will-not* transpose the matrices, so the caller is responsible for
  // ensuring that the matrices are in the correct format and have the correct
  // dimensions.
  void matmul(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
              cudaDataType_t dtype) {
    // CUDA is column-major, while triton is row-major, therefore we need to
    // reverse the order of the matrices ( A * B = (B^T * A^T)^T ).
    gemm_impl(n, m, k, B, A, 0, C, dtype, 1.0f, 0.0f);
  }

  void gemm(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C, uint64_t D,
            cudaDataType_t dtype, float alpha, float beta) {
    gemm_impl(n, m, k, B, A, C, D, dtype, alpha, beta);
  }

  void block_scaled_matmul_mxfp8(int m, int n, int k, uint64_t A, uint64_t B,
                                 uint64_t D_out, uint64_t scale_A,
                                 uint64_t scale_B) {
    block_scaled_matmul(m, n, k, A, B, D_out, scale_A, scale_B, true);
  }

  void block_scaled_matmul_nvfp4(int m, int n, int k, uint64_t A, uint64_t B,
                                 uint64_t D_out, uint64_t scale_A,
                                 uint64_t scale_B) {
    block_scaled_matmul(m, n, k, A, B, D_out, scale_A, scale_B, false);
  }
};

#endif // TRITON_CUBLAS_INSTANCE_H
