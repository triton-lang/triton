#include "cutlass/library/handle.h"
#include "cutlass/library/library.h"
#include "cutlass/library/operation_table.h"
#include "cutlass/library/singleton.h"
#include "pybind11/pybind11.h"
#include "triton/tools/bench.hpp"

using namespace cutlass;
using namespace cutlass::library;

std::map<std::vector<size_t>, const Operation *> op_cache_;

static int const kHostWorkspaceSize = (4 << 10);
static int const kDeviceWorkspaceSize = (4 << 20);

void run(int M, int N, int K,
         int lda, int ldb, int ldc, int ldd,
         void const *ptr_A, void const *ptr_B, void const *ptr_C, void *ptr_D,
         void const *alpha, void const *beta,
         ScalarPointerMode scalar_mode,
         const Operation *operation,
         cudaStream_t stream) {

  GemmUniversalConfiguration configuration{
      GemmUniversalMode::kGemm,
      {M, N, K},
      1,
      lda,
      ldb,
      ldc,
      ldd};

  // host workspace size
  uint64_t host_workspace_size_needed = operation->get_host_workspace_size(&configuration);
  if (uint64_t(kHostWorkspaceSize) < host_workspace_size_needed)
    throw std::runtime_error("Unable to find gemm operation");
  char host_workspace[kHostWorkspaceSize];

  // device workspace size
  uint64_t device_workspace_size_needed = operation->get_device_workspace_size(&configuration);
  if (uint64_t(kDeviceWorkspaceSize) < device_workspace_size_needed)
    throw std::runtime_error("Unable to find gemm operation");
  static void *device_workspace;

  // Initialize host and device workspaces
  Status status = operation->initialize(&configuration, host_workspace, device_workspace, stream);
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error("Unable to initialize workspace");

  // Run the operator
  GemmArguments arguments{ptr_A, ptr_B, ptr_C, ptr_D, alpha, beta, scalar_mode};
  operation->run(&arguments, host_workspace, device_workspace, stream);
}

const Operation *autotune(int M, int N, int K,
                          NumericTypeID element_compute,
                          NumericTypeID element_scalar,
                          void const *alpha,
                          NumericTypeID element_A,
                          LayoutTypeID layout_A,
                          ComplexTransform transform_A,
                          void const *ptr_A,
                          int lda,
                          NumericTypeID element_B,
                          LayoutTypeID layout_B,
                          ComplexTransform transform_B,
                          void const *ptr_B,
                          int ldb,
                          void const *beta,
                          NumericTypeID element_C,
                          void const *ptr_C,
                          int ldc,
                          void *ptr_D,
                          int ldd,
                          ScalarPointerMode scalar_mode,
                          int device_id,
                          cudaStream_t stream) {

  // index operation table with functional key
  GemmFunctionalKey key(
      Provider::kCUTLASS,
      GemmKind::kUniversal,
      element_compute,
      element_scalar,
      element_A,
      layout_A,
      transform_A,
      element_B,
      layout_B,
      transform_B,
      element_C);

  auto operators_it = Singleton::get().operation_table.gemm_operations.find(key);
  if (operators_it == Singleton::get().operation_table.gemm_operations.end())
    throw std::runtime_error("Unable to find gemm operation");
  if (operators_it->second.empty())
    throw std::runtime_error("Unable to find gemm operation");

  cudaDeviceProp device_prop;
  cudaError_t error = cudaGetDeviceProperties(&device_prop, device_id);
  if (error != cudaSuccess)
    throw std::runtime_error("Unable to get device properties");
  int cc = device_prop.major * 10 + device_prop.minor;

  // index operation table with preference key
  // assume 8-bytes aligned memory pointers
  int alignment = 8;
  GemmPreferenceKey preference_key(cc, alignment);
  auto autotune_it = operators_it->second.find(preference_key);
  if (autotune_it == operators_it->second.end())
    throw std::runtime_error("Unable to find gemm operation");
  const std::vector<const Operation *> &operations = autotune_it->second;
  if (operations.empty())
    throw std::runtime_error("Unable to find gemm operation");

  // auto-tune
  const Operation *best = nullptr;
  double best_ms = std::numeric_limits<double>::max();
  for (const Operation *op : operations) {
    auto fn = [&]() { run(M, N, K, lda, ldb, ldc, ldd, ptr_A, ptr_B, ptr_C, ptr_D,
                          alpha, beta, scalar_mode, op, stream); };
    triton::driver::cu_stream tt_stream((CUstream)stream, false);
    double ms = triton::tools::bench(fn, &tt_stream, 10, 25);
    if (ms < best_ms) {
      best_ms = ms;
      best = op;
    }
  }
  return best;
}

// map of torch datatypes to cutlass datatypes
std::map<std::string, NumericTypeID> type_map = {
    {"float16", NumericTypeID::kF16},
    {"float32", NumericTypeID::kF32},
    {"float64", NumericTypeID::kF64}};

void cutlass_matmul(uintptr_t A, uintptr_t B, uintptr_t C,
                    size_t M, size_t N, size_t K,
                    size_t stride_a_0, size_t stride_a_1,
                    size_t stride_b_0, size_t stride_b_1,
                    size_t stride_c_0, size_t stride_c_1,
                    std::string type_a, std::string type_b, std::string type_c,
                    size_t dev_id, uint64_t stream_handle) {
  void *ptr_A = (void *)A;
  void *ptr_B = (void *)B;
  void *ptr_C = (void *)C;
  void *ptr_D = ptr_C;
  size_t lda = stride_a_0;
  size_t ldb = stride_b_0;
  size_t ldc = stride_c_1;
  size_t ldd = ldc;
  float alpha = 1.0f;
  float beta = 0.0f;
  // layout for A
  LayoutTypeID layout_A;
  if (stride_a_0 == 1)
    layout_A = LayoutTypeID::kColumnMajor;
  else if (stride_a_1 == 1)
    layout_A = LayoutTypeID::kRowMajor;
  else
    throw std::runtime_error("A layout is not supported");
  // layout for B
  LayoutTypeID layout_B;
  if (stride_b_0 == 1)
    layout_B = LayoutTypeID::kColumnMajor;
  else if (stride_b_1 == 1)
    layout_B = LayoutTypeID::kRowMajor;
  else
    throw std::runtime_error("B layout is not supported");
  // data types
  NumericTypeID element_compute = NumericTypeID::kF32;
  NumericTypeID element_A = type_map[type_a];
  NumericTypeID element_B = type_map[type_b];
  NumericTypeID element_C = type_map[type_c];
  // misc. flags
  ScalarPointerMode scalar_mode = ScalarPointerMode::kHost;
  NumericTypeID element_scalar = NumericTypeID::kF32;
  ComplexTransform transform_A = ComplexTransform::kNone;
  ComplexTransform transform_B = ComplexTransform::kNone;
  // runtime flags
  cudaStream_t stream = (cudaStream_t)stream_handle;
  // auto-tune
  std::vector<size_t> tune_key = {M, N, K, (size_t)element_A, (size_t)element_B, (size_t)element_C,
                                  dev_id, (size_t)element_compute, (size_t)scalar_mode};
  auto it = op_cache_.find(tune_key);
  if (it == op_cache_.end()) {
    const Operation *op = autotune(M, N, K, element_compute, element_scalar, &alpha,
                                   element_A, layout_A, transform_A, ptr_A, lda,
                                   element_B, layout_B, transform_B, ptr_B, ldb,
                                   &beta, element_C, ptr_C, ldc, ptr_D, ldd, scalar_mode,
                                   dev_id, stream);
    it = op_cache_.insert({tune_key, op}).first;
  }
  run(M, N, K, lda, ldb, ldc, ldd, ptr_A, ptr_B, ptr_C, ptr_D, &alpha, &beta,
      scalar_mode, it->second, stream);
}

void init_cutlass(pybind11::module &m) {
  pybind11::module subm = m.def_submodule("cutlass");
  subm.def("matmul", &cutlass_matmul, "matrix multiplication");
}