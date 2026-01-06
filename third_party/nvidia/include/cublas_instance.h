#ifndef TRITON_CUBLAS_INSTANCE_H
#define TRITON_CUBLAS_INSTANCE_H

#include "cublas_types.h"
#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

  // Typedefs for CUDA runtime functions (for autotuning timing)
  typedef cudaError_t (*cudaStreamCreate_t)(cudaStream_t *);
  typedef cudaError_t (*cudaStreamDestroy_t)(cudaStream_t);
  typedef cudaError_t (*cudaEventCreate_t)(cudaEvent_t *);
  typedef cudaError_t (*cudaEventDestroy_t)(cudaEvent_t);
  typedef cudaError_t (*cudaEventRecord_t)(cudaEvent_t, cudaStream_t);
  typedef cudaError_t (*cudaEventSynchronize_t)(cudaEvent_t);
  typedef cudaError_t (*cudaEventElapsedTime_t)(float *, cudaEvent_t,
                                                cudaEvent_t);
  typedef cudaError_t (*cudaMalloc_t)(void **, size_t);
  typedef cudaError_t (*cudaFree_t)(void *);
  typedef cudaError_t (*cudaMemsetAsync_t)(void *, int, size_t, cudaStream_t);
  typedef cudaError_t (*cudaStreamSynchronize_t)(cudaStream_t);
  typedef cudaError_t (*cudaDeviceGetAttribute_t)(int *, cudaDeviceAttr, int);
  typedef cudaError_t (*cudaGetDevice_t)(int *);

  static constexpr const char *name = "libcublas.so";
  static constexpr const char *cudartName = "libcudart.so";

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

  // CUDA runtime function pointers
  cudaStreamCreate_t cudaStreamCreateFn;
  cudaStreamDestroy_t cudaStreamDestroyFn;
  cudaEventCreate_t cudaEventCreateFn;
  cudaEventDestroy_t cudaEventDestroyFn;
  cudaEventRecord_t cudaEventRecordFn;
  cudaEventSynchronize_t cudaEventSynchronizeFn;
  cudaEventElapsedTime_t cudaEventElapsedTimeFn;
  cudaMalloc_t cudaMallocFn;
  cudaFree_t cudaFreeFn;
  cudaMemsetAsync_t cudaMemsetAsyncFn;
  cudaStreamSynchronize_t cudaStreamSynchronizeFn;
  cudaDeviceGetAttribute_t cudaDeviceGetAttributeFn;
  cudaGetDevice_t cudaGetDeviceFn;

  // L2 cache flush buffer (allocated once, reused for autotuning)
  void *l2FlushBuffer = nullptr;
  size_t l2FlushSize = 0;

  // Autotuning enabled flag (controlled by TRITON_CUBLASLT_AUTOTUNE env var)
  bool autotuneEnabled = false;

  void *dylibHandle = nullptr;
  void *cudartHandle = nullptr;
  cublasLtHandle_t ltHandle;

  void *workspace = nullptr;
  size_t workspaceSize = 0;

  cublasLtMatmulPreference_t preference = NULL;

  // Autotuning parameters
  static constexpr int kRequestedAlgoCount = 8;
  static constexpr int kCalibrationRuns = 10;
  static constexpr float kTargetBenchmarkTimeMs = 100.0f; // 0.1 seconds

  // Cache key for autotuned algorithms
  struct AlgoCacheKey {
    int m, n, k;
    cudaDataType_t dtype;
    float alpha, beta;
    const char *opName;

    bool operator==(const AlgoCacheKey &other) const {
      return m == other.m && n == other.n && k == other.k &&
             dtype == other.dtype && alpha == other.alpha &&
             beta == other.beta &&
             opName == other.opName; // pointer comparison is fine for literals
    }
  };

  struct AlgoCacheKeyHash {
    size_t operator()(const AlgoCacheKey &key) const {
      size_t h = std::hash<int>()(key.m);
      h ^= std::hash<int>()(key.n) << 1;
      h ^= std::hash<int>()(key.k) << 2;
      h ^= std::hash<int>()(static_cast<int>(key.dtype)) << 3;
      h ^= std::hash<float>()(key.alpha) << 4;
      h ^= std::hash<float>()(key.beta) << 5;
      h ^= std::hash<const void *>()(key.opName) << 6;
      return h;
    }
  };

  // Cache: maps problem config to best algorithm
  std::unordered_map<AlgoCacheKey, cublasLtMatmulAlgo_t, AlgoCacheKeyHash>
      algoCache;

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

  void loadCudartDylib() {
    if (cudartHandle == nullptr) {
      cudartHandle = dlopen(cudartName, RTLD_NOLOAD);
    }
    if (cudartHandle == nullptr) {
      cudartHandle = dlopen(cudartName, RTLD_LOCAL | RTLD_LAZY);
    }
    if (cudartHandle == nullptr) {
      throw std::runtime_error("Could not find `" + std::string(cudartName) +
                               "`. Make sure it is in your LD_LIBRARY_PATH.");
    }
    dlerror();

    cudaStreamCreateFn =
        (cudaStreamCreate_t)dlsym(cudartHandle, "cudaStreamCreate");
    cudaStreamDestroyFn =
        (cudaStreamDestroy_t)dlsym(cudartHandle, "cudaStreamDestroy");
    cudaEventCreateFn =
        (cudaEventCreate_t)dlsym(cudartHandle, "cudaEventCreate");
    cudaEventDestroyFn =
        (cudaEventDestroy_t)dlsym(cudartHandle, "cudaEventDestroy");
    cudaEventRecordFn =
        (cudaEventRecord_t)dlsym(cudartHandle, "cudaEventRecord");
    cudaEventSynchronizeFn =
        (cudaEventSynchronize_t)dlsym(cudartHandle, "cudaEventSynchronize");
    cudaEventElapsedTimeFn =
        (cudaEventElapsedTime_t)dlsym(cudartHandle, "cudaEventElapsedTime");
    cudaMallocFn = (cudaMalloc_t)dlsym(cudartHandle, "cudaMalloc");
    cudaFreeFn = (cudaFree_t)dlsym(cudartHandle, "cudaFree");
    cudaMemsetAsyncFn =
        (cudaMemsetAsync_t)dlsym(cudartHandle, "cudaMemsetAsync");
    cudaStreamSynchronizeFn =
        (cudaStreamSynchronize_t)dlsym(cudartHandle, "cudaStreamSynchronize");
    cudaDeviceGetAttributeFn =
        (cudaDeviceGetAttribute_t)dlsym(cudartHandle, "cudaDeviceGetAttribute");
    cudaGetDeviceFn = (cudaGetDevice_t)dlsym(cudartHandle, "cudaGetDevice");

    const char *dlsym_error = dlerror();
    if (dlsym_error) {
      throw std::runtime_error("Could not load symbol from `" +
                               std::string(cudartName) +
                               "`: " + std::string(dlsym_error));
    }

    // Query L2 cache size from current device and allocate flush buffer
    int device = 0;
    int l2CacheSize = 0;
    cudaGetDeviceFn(&device);
    cudaDeviceGetAttributeFn(&l2CacheSize, cudaDevAttrL2CacheSize, device);
    // Use L2 size
    l2FlushSize = static_cast<size_t>(l2CacheSize);
    cudaMallocFn(&l2FlushBuffer, l2FlushSize);
  }

  void unloadCublasDylib() {
    if (l2FlushBuffer && cudaFreeFn)
      cudaFreeFn(l2FlushBuffer);
    if (dylibHandle)
      dlclose(dylibHandle);
    if (cudartHandle)
      dlclose(cudartHandle);
  }

  void successOrExit(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cuBLAS Error: " + std::to_string(status) +
                               "\n");
    }
  }

  float computeMedian(std::vector<float> &times) {
    const size_t size = times.size();
    if (size == 0)
      return 0;
    std::sort(times.begin(), times.end());
    const size_t mid = size / 2;
    return (size % 2 == 0) ? (times[mid] + times[mid - 1]) / 2 : times[mid];
  }

  // Autotune and execute matmul - shared implementation for all matmul types
  // When TRITON_CUBLASLT_AUTOTUNE=1, caches best algorithm per config
  // Otherwise, uses the first heuristic result (default cuBLAS behavior)
  // Autotuning was implemented with the following as reference implementation:
  // clang-format off
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemmSimpleAutoTuning/sample_cublasLt_LtSgemmSimpleAutoTuning.cu
  // clang-format on
  void autotuneAndExecute(cublasLtMatmulDesc_t matmulDesc,
                          cublasLtMatrixLayout_t Adesc,
                          cublasLtMatrixLayout_t Bdesc,
                          cublasLtMatrixLayout_t Cdesc,
                          cublasLtMatrixLayout_t Ddesc, const float *alpha,
                          const void *A, const void *B, const float *beta,
                          const void *C, void *D, int m, int n, int k,
                          cudaDataType_t dtype, const char *opName) {
    // When autotuning is disabled, just use the first heuristic result
    if (!autotuneEnabled) {
      cublasLtMatmulHeuristicResult_t heuristicResult = {};
      int returnedResults = 0;
      successOrExit(cublasLtMatmulAlgoGetHeuristic(
          ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
          &heuristicResult, &returnedResults));
      if (returnedResults == 0) {
        throw std::runtime_error(
            "No valid algorithm found by cublasLtMatmulAlgoGetHeuristic");
      }
      successOrExit(cublasLtMatmul(
          ltHandle, matmulDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D,
          Ddesc, &heuristicResult.algo, workspace, workspaceSize, 0));
      return;
    }

    // Autotuning enabled - check cache first
    AlgoCacheKey cacheKey{m, n, k, dtype, *alpha, *beta, opName};
    auto it = algoCache.find(cacheKey);
    if (it != algoCache.end()) {
      // Use cached algorithm directly
      successOrExit(cublasLtMatmul(ltHandle, matmulDesc, alpha, A, Adesc, B,
                                   Bdesc, beta, C, Cdesc, D, Ddesc, &it->second,
                                   workspace, workspaceSize, 0));
      return;
    }

    // First time seeing this configuration - run autotuning
    cublasLtMatmulHeuristicResult_t heuristicResults[kRequestedAlgoCount] = {};
    int returnedResults = 0;

    successOrExit(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, preference,
        kRequestedAlgoCount, heuristicResults, &returnedResults));

    if (returnedResults == 0) {
      throw std::runtime_error(
          "No valid algorithm found by cublasLtMatmulAlgoGetHeuristic");
    }

    cudaStream_t stream;
    cudaEvent_t startEvent, stopEvent;
    cudaStreamCreateFn(&stream);
    cudaEventCreateFn(&startEvent);
    cudaEventCreateFn(&stopEvent);

    int bestAlgoIdx = 0;
    float bestAlgoTime = 0;
    float baselineAlgoTime = 0;

    for (int algoIdx = 0; algoIdx < returnedResults; algoIdx++) {
      // Phase 1: Calibration - run a few iterations to estimate kernel time
      float calibrationTotalTime = 0;
      for (int calibIdx = 0; calibIdx < kCalibrationRuns; calibIdx++) {
        cudaMemsetAsyncFn(l2FlushBuffer, 0, l2FlushSize, stream);
        cudaStreamSynchronizeFn(stream);

        cudaEventRecordFn(startEvent, stream);

        cublasLtMatmul(ltHandle, matmulDesc, alpha, A, Adesc, B, Bdesc, beta, C,
                       Cdesc, D, Ddesc, &heuristicResults[algoIdx].algo,
                       workspace, workspaceSize, stream);

        cudaEventRecordFn(stopEvent, stream);
        cudaEventSynchronizeFn(stopEvent);

        float time;
        cudaEventElapsedTimeFn(&time, startEvent, stopEvent);
        calibrationTotalTime += time;
      }

      // Estimate average time per iteration and compute iterations needed
      float avgTimePerIter = calibrationTotalTime / kCalibrationRuns;
      int benchmarkRuns = std::max(
          1, static_cast<int>(kTargetBenchmarkTimeMs / avgTimePerIter));

      // Phase 2: Benchmark with calculated iteration count
      std::vector<float> algoTimes(benchmarkRuns);
      for (int checkIdx = 0; checkIdx < benchmarkRuns; checkIdx++) {
        // Flush L2 cache before timing to get consistent measurements
        cudaMemsetAsyncFn(l2FlushBuffer, 0, l2FlushSize, stream);
        cudaStreamSynchronizeFn(stream);

        cudaEventRecordFn(startEvent, stream);

        cublasLtMatmul(ltHandle, matmulDesc, alpha, A, Adesc, B, Bdesc, beta, C,
                       Cdesc, D, Ddesc, &heuristicResults[algoIdx].algo,
                       workspace, workspaceSize, stream);

        cudaEventRecordFn(stopEvent, stream);
        cudaEventSynchronizeFn(stopEvent);

        float time;
        cudaEventElapsedTimeFn(&time, startEvent, stopEvent);
        algoTimes[checkIdx] = time;
      }

      float medianTime = computeMedian(algoTimes);
      if (algoIdx == 0) {
        bestAlgoTime = medianTime;
        bestAlgoIdx = algoIdx;
        baselineAlgoTime = medianTime;
      } else if (medianTime < (bestAlgoTime / 1.02f)) {
        // Only update if the new algorithm is at least 2% faster than the best
        // algorithm This is because of measurement noise, which typically can
        // hover around 1-2%
        bestAlgoTime = medianTime;
        bestAlgoIdx = algoIdx;
      }
    }

    // Cache the best algorithm for future calls
    algoCache[cacheKey] = heuristicResults[bestAlgoIdx].algo;

    // Debug output - comment in as needed
    // std::cout << "[cuBLAS Autotune] " << opName << " (m=" << m << ", n=" << n
    //           << ", k=" << k << "): selected algo " << bestAlgoIdx << "/"
    //           << returnedResults << " with median time "
    //           << baselineAlgoTime / bestAlgoTime << "x faster than baseline"
    //           << std::endl;

    // Final execution with best algorithm (on default stream)
    successOrExit(cublasLtMatmul(ltHandle, matmulDesc, alpha, A, Adesc, B,
                                 Bdesc, beta, C, Cdesc, D, Ddesc,
                                 &heuristicResults[bestAlgoIdx].algo, workspace,
                                 workspaceSize, 0));

    cudaEventDestroyFn(stopEvent);
    cudaEventDestroyFn(startEvent);
    cudaStreamDestroyFn(stream);
  }

  // Simple wrapper around the cublasLtMatmul function with autotuning
  void gemm_impl(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
                 uint64_t D, cudaDataType_t dtype, float alpha, float beta) {
    cublasLtMatmulDesc_t matmulDesc = NULL;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    int8_t fastAccum = 1;

    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL,
                           Ddesc = NULL;

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

    // Autotune and execute with best algorithm
    autotuneAndExecute(matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, &alpha,
                       (void *)A, (void *)B, &beta, (void *)C, (void *)D, m, n,
                       k, dtype, "gemm");

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

    // Autotune and execute (B and A swapped for row-major to col-major)
    const char *opName =
        is_mxfp8 ? "block_scaled_matmul_mxfp8" : "block_scaled_matmul_nvfp4";
    autotuneAndExecute(matmulDesc, Bdesc, Adesc, Cdesc, Cdesc, &alpha,
                       (void *)B, (void *)A, &beta, (void *)D_out,
                       (void *)D_out, m, n, k, dataType, opName);

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
    loadCudartDylib();
    cublasLtCreate(&ltHandle);

    successOrExit(cublasLtMatmulPreferenceCreate(&preference));
    successOrExit(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));

    // Check if autotuning is enabled via environment variable
    const char *autotuneEnv = std::getenv("TRITON_CUBLASLT_AUTOTUNE");
    autotuneEnabled = autotuneEnv != nullptr && std::string(autotuneEnv) == "1";
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
