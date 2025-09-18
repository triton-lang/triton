#ifndef TRITON_HIPBLAS_INSTANCE_H
#define TRITON_HIPBLAS_INSTANCE_H

#include "hipblas_types.h"
#include <dlfcn.h>
#include <sstream>
#include <stdexcept>
#include <string>

// this gets translated to rocblastlt_compute_f32_fast_f8 internally by
// hipblasLt
constexpr int HIPBLAS_COMPUTE_32F_FAST_F8 = 104;
constexpr int HIPBLAS_COMPUTE_32F_FAST_FBF_OCP = 105;

class HipblasLtInstance {
  // Typedefs for hipblas functions
  typedef hipblasStatus_t (*hipblasLtCreate_t)(hipblasLtHandle_t *);
  typedef hipblasStatus_t (*hipblasLtDestroy_t)(hipblasLtHandle_t);
  typedef hipblasStatus_t (*hipblasLtMatmulDescCreate_t)(
      hipblasLtMatmulDesc_t *, hipblasComputeType_t, hipDataType);
  typedef hipblasStatus_t (*hipblasLtMatmulDescDestroy_t)(
      hipblasLtMatmulDesc_t);
  typedef hipblasStatus_t (*hipblasLtMatmulDescSetAttribute_t)(
      hipblasLtMatmulDesc_t, hipblasLtMatmulDescAttributes_t, const void *,
      size_t);
  typedef hipblasStatus_t (*hipblasLtMatrixLayoutCreate_t)(
      hipblasLtMatrixLayout_t *, hipDataType, uint64_t, uint64_t, int64_t);
  typedef hipblasStatus_t (*hipblasLtMatrixLayoutDestroy_t)(
      hipblasLtMatrixLayout_t);
  typedef hipblasStatus_t (*hipblasLtMatmulPreferenceCreate_t)(
      hipblasLtMatmulPreference_t *);
  typedef hipblasStatus_t (*hipblasLtMatmulPreferenceDestroy_t)(
      hipblasLtMatmulPreference_t);
  typedef hipblasStatus_t (*hipblasLtMatmulPreferenceSetAttribute_t)(
      hipblasLtMatmulPreference_t, hipblasLtMatmulPreferenceAttributes_t,
      const void *, size_t);
  typedef hipblasStatus_t (*hipblasLtMatmulAlgoGetHeuristic_t)(
      hipblasLtHandle_t, hipblasLtMatmulDesc_t, hipblasLtMatrixLayout_t,
      hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t,
      hipblasLtMatmulPreference_t, int, hipblasLtMatmulHeuristicResult_t *,
      int *);
  typedef hipblasStatus_t (*hipblasLtMatmul_t)(
      hipblasLtHandle_t, hipblasLtMatmulDesc_t, const void *, const void *,
      const hipblasLtMatrixLayout_t, const void *,
      const hipblasLtMatrixLayout_t, const void *, const void *,
      const hipblasLtMatrixLayout_t, void *, const hipblasLtMatrixLayout_t,
      const hipblasLtMatmulAlgo_t *, void *, size_t, hipStream_t);

  static constexpr const char *name = "libhipblaslt.so";

  hipblasLtCreate_t hipblasLtCreate;
  hipblasLtDestroy_t hipblasLtDestroy;
  hipblasLtMatmulDescCreate_t hipblasLtMatmulDescCreate;
  hipblasLtMatmulDescDestroy_t hipblasLtMatmulDescDestroy;
  hipblasLtMatmulDescSetAttribute_t hipblasLtMatmulDescSetAttribute;
  hipblasLtMatrixLayoutCreate_t hipblasLtMatrixLayoutCreate;
  hipblasLtMatrixLayoutDestroy_t hipblasLtMatrixLayoutDestroy;
  hipblasLtMatmulPreferenceCreate_t hipblasLtMatmulPreferenceCreate;
  hipblasLtMatmulPreferenceDestroy_t hipblasLtMatmulPreferenceDestroy;
  hipblasLtMatmulPreferenceSetAttribute_t hipblasLtMatmulPreferenceSetAttribute;
  hipblasLtMatmulAlgoGetHeuristic_t hipblasLtMatmulAlgoGetHeuristic;
  hipblasLtMatmul_t hipblasLtMatmul;

  void *dylibHandle = nullptr;
  hipblasLtHandle_t ltHandle;

  void *workspace = nullptr;
  size_t workspaceSize = 0;

  hipblasLtMatmulPreference_t preference = NULL;

  void loadHipBlasDylib() {
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

    hipblasLtCreate = (hipblasLtCreate_t)dlsym(dylibHandle, "hipblasLtCreate");
    hipblasLtDestroy =
        (hipblasLtDestroy_t)dlsym(dylibHandle, "hipblasLtDestroy");
    hipblasLtMatmulDescCreate = (hipblasLtMatmulDescCreate_t)dlsym(
        dylibHandle, "hipblasLtMatmulDescCreate");
    hipblasLtMatmulDescDestroy = (hipblasLtMatmulDescDestroy_t)dlsym(
        dylibHandle, "hipblasLtMatmulDescDestroy");
    hipblasLtMatmulDescSetAttribute = (hipblasLtMatmulDescSetAttribute_t)dlsym(
        dylibHandle, "hipblasLtMatmulDescSetAttribute");
    hipblasLtMatrixLayoutCreate = (hipblasLtMatrixLayoutCreate_t)dlsym(
        dylibHandle, "hipblasLtMatrixLayoutCreate");
    hipblasLtMatrixLayoutDestroy = (hipblasLtMatrixLayoutDestroy_t)dlsym(
        dylibHandle, "hipblasLtMatrixLayoutDestroy");
    hipblasLtMatmulPreferenceCreate = (hipblasLtMatmulPreferenceCreate_t)dlsym(
        dylibHandle, "hipblasLtMatmulPreferenceCreate");
    hipblasLtMatmulPreferenceDestroy =
        (hipblasLtMatmulPreferenceDestroy_t)dlsym(
            dylibHandle, "hipblasLtMatmulPreferenceDestroy");
    hipblasLtMatmulPreferenceSetAttribute =
        (hipblasLtMatmulPreferenceSetAttribute_t)dlsym(
            dylibHandle, "hipblasLtMatmulPreferenceSetAttribute");
    hipblasLtMatmulAlgoGetHeuristic = (hipblasLtMatmulAlgoGetHeuristic_t)dlsym(
        dylibHandle, "hipblasLtMatmulAlgoGetHeuristic");
    hipblasLtMatmul = (hipblasLtMatmul_t)dlsym(dylibHandle, "hipblasLtMatmul");

    const char *dlsym_error = dlerror();
    if (dlsym_error) {
      throw std::runtime_error("Could not load symbol from `" +
                               std::string(name) +
                               "`: " + std::string(dlsym_error));
    }
  }

  void unloadHipBlasDylib() { dlclose(dylibHandle); }

  void successOrExit(hipblasStatus_t status, const std::string &context = "") {
    if (status != HIPBLAS_STATUS_SUCCESS) {
      std::ostringstream oss;
      oss << "HIPBLAS Error in " << context << ": " << status;

      switch (status) {
      case HIPBLAS_STATUS_NOT_INITIALIZED:
        oss << " (NOT_INITIALIZED)";
        break;
      case HIPBLAS_STATUS_ALLOC_FAILED:
        oss << " (ALLOC_FAILED)";
        break;
      case HIPBLAS_STATUS_INVALID_VALUE:
        oss << " (INVALID_VALUE - Parameters are unexpectedly NULL, in "
               "conflict or in impossible configuration)";
        break;
      case HIPBLAS_STATUS_MAPPING_ERROR:
        oss << " (MAPPING_ERROR)";
        break;
      case HIPBLAS_STATUS_EXECUTION_FAILED:
        oss << " (EXECUTION_FAILED)";
        break;
      case HIPBLAS_STATUS_INTERNAL_ERROR:
        oss << " (INTERNAL_ERROR)";
        break;
      case HIPBLAS_STATUS_NOT_SUPPORTED:
        oss << " (NOT_SUPPORTED)";
        break;
      case HIPBLAS_STATUS_INVALID_ENUM:
        oss << " (INVALID_ENUM)";
        break;
      case HIPBLAS_STATUS_UNKNOWN:
        oss << " (UNKNOWN)";
        break;
      default:
        oss << " (UNKNOWN_ERROR_CODE)";
      }

      throw std::runtime_error(oss.str() + "\n");
    }
  }

  void gemm_impl(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
                 uint64_t D, hipDataType dtype, float alpha, float beta) {

    hipblasLtMatmulDesc_t matmulDesc = NULL;

    hipblasOperation_t transa = HIPBLAS_OP_T;
    hipblasOperation_t transb = HIPBLAS_OP_N;

    int8_t fastAccum = 1;

    hipblasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL,
                            Ddesc = NULL;
    auto isFP8 = dtype == HIP_R_8F_E4M3 || dtype == HIP_R_8F_E5M2 ||
                 dtype == HIP_R_8F_E4M3_FNUZ || dtype == HIP_R_8F_E5M2_FNUZ;
    int returnedResults = 0;
    hipblasLtMatmulHeuristicResult_t heuristicResult = {};

    hipblasComputeType_t computeType;
    if (dtype == HIP_R_8F_E4M3) {
      computeType = (hipblasComputeType_t)HIPBLAS_COMPUTE_32F_FAST_F8;
    } else if (dtype == HIP_R_8F_E5M2) {
      computeType = (hipblasComputeType_t)HIPBLAS_COMPUTE_32F_FAST_FBF_OCP;
    } else if (dtype == HIP_R_8F_E4M3_FNUZ) {
      computeType = HIPBLAS_COMPUTE_32F_FAST_8F_FNUZ;
    } else if (dtype == HIP_R_8F_E5M2_FNUZ) {
      computeType = HIPBLAS_COMPUTE_32F_FAST_8BF_FNUZ;
    } else if (dtype == HIP_R_32F) {
      computeType = HIPBLAS_COMPUTE_32F_FAST_TF32;
    } else {
      computeType = HIPBLAS_COMPUTE_32F;
    }
    successOrExit(
        hipblasLtMatmulDescCreate(&matmulDesc, computeType, HIP_R_32F));
    successOrExit(hipblasLtMatmulDescSetAttribute(
        matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    successOrExit(hipblasLtMatmulDescSetAttribute(
        matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    if (isFP8) {
      hipDataType a_in = dtype;
      hipDataType b_in = dtype;
      successOrExit(hipblasLtMatmulDescSetAttribute(
          matmulDesc, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &a_in,
          sizeof(a_in)));
      successOrExit(hipblasLtMatmulDescSetAttribute(
          matmulDesc, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &b_in,
          sizeof(b_in)));
    }

    auto c_dtype = (isFP8) ? HIP_R_16F : dtype;

    successOrExit(hipblasLtMatrixLayoutCreate(&Adesc, dtype, k, m, k));
    successOrExit(hipblasLtMatrixLayoutCreate(&Bdesc, dtype, k, n, k));
    successOrExit(hipblasLtMatrixLayoutCreate(&Cdesc, c_dtype, m, n, m));

    successOrExit(hipblasLtMatrixLayoutCreate(&Ddesc, c_dtype, m, n, m));
    successOrExit(hipblasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
        &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
      std::ostringstream oss;
      oss << "No valid algorithm found by hipblasLtMatmulAlgoGetHeuristic\n";
      oss << "Matrix details:\n";
      oss << "  A: dtype=" << dtype << ", shape=(" << m << "," << k
          << "), leading_dim=" << m << "\n";
      oss << "  B: dtype=" << dtype << ", shape=(" << n << "," << k
          << "), leading_dim=" << n << " (will be transposed)\n";
      oss << "  C: dtype=" << c_dtype << ", shape=(" << m << "," << n
          << "), leading_dim=" << m << "\n";
      oss << "  D: dtype=" << c_dtype << ", shape=(" << m << "," << n
          << "), leading_dim=" << m << "\n";
      oss << "  Compute type: " << computeType << "\n";
      oss << "  Requested algorithms: 1\n";
      throw std::runtime_error(oss.str());
    }

    if (heuristicResult.workspaceSize > workspaceSize) {
      throw std::runtime_error("Insufficient workspace: need " +
                               std::to_string(heuristicResult.workspaceSize) +
                               " bytes, have " + std::to_string(workspaceSize) +
                               " bytes");
    }

    successOrExit(hipblasLtMatmul(
        ltHandle, matmulDesc, &alpha, (void *)A, Adesc, (void *)B, Bdesc, &beta,
        (void *)C, Cdesc, (void *)D, Ddesc, &heuristicResult.algo,
        (void *)workspace, workspaceSize, 0));

    if (Ddesc)
      successOrExit(hipblasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc)
      successOrExit(hipblasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
      successOrExit(hipblasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
      successOrExit(hipblasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc)
      successOrExit(hipblasLtMatmulDescDestroy(matmulDesc));
  }

public:
  HipblasLtInstance(uint64_t workspace, size_t workspaceSize)
      : workspace((void *)workspace), workspaceSize(workspaceSize) {
    loadHipBlasDylib();
    successOrExit(hipblasLtCreate(&ltHandle));
    successOrExit(hipblasLtMatmulPreferenceCreate(&preference));
    successOrExit(hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));
  }
  ~HipblasLtInstance() {
    if (preference)
      successOrExit(hipblasLtMatmulPreferenceDestroy(preference));

    hipblasLtDestroy(ltHandle);
    unloadHipBlasDylib();
  }

  void matmul(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
              hipDataType dtype) {
    // HIP is column-major, while triton is row-major, therefore we need to
    // reverse the order of the matrices ( A * B = (B^T * A^T)^T ).
    // Note: HipBLAS requires a valid C pointer even when beta=0, so we pass C
    // instead of 0
    gemm_impl(n, m, k, B, A, C, C, dtype, 1.0f, 0.0f);
  }

  void gemm(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C, uint64_t D,
            hipDataType dtype, float alpha, float beta) {
    gemm_impl(n, m, k, B, A, C, D, dtype, alpha, beta);
  }
};
#endif // TRITON_HIPBLAS_INSTANCE_H
