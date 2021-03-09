#include "cutlass/library/handle.h"
#include "cutlass/library/library.h"
#include "cutlass/library/operation_table.h"
#include "cutlass/library/singleton.h"
#include "pybind11/pybind11.h"
#include "triton/tools/bench.hpp"
#include <torch/extension.h>

using namespace cutlass;
using namespace cutlass::library;

typedef std::tuple<GemmFunctionalKey, GemmPreferenceKey, int, int, int> tune_key_t;

struct hash {
  size_t operator()(const tune_key_t &t) const {
    size_t seed = 0;
    auto update_seed = [&](int v) {
      seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    };
    update_seed(GemmFunctionalKeyHasher()(std::get<0>(t)));
    update_seed(std::hash<int>()(std::get<1>(t).compute_capability));
    update_seed(std::hash<int>()(std::get<1>(t).alignment));
    update_seed(std::hash<int>()(std::get<2>(t)));
    update_seed(std::hash<int>()(std::get<3>(t)));
    update_seed(std::hash<int>()(std::get<4>(t)));
  }
};

std::unordered_map<tune_key_t, const Operation *, hash>
    op_cache;

static int const kHostWorkspaceSize = (4 << 10);
static int const kDeviceWorkspaceSize = (4 << 20);

void run(int M, int N, int K,
         int lda, int ldb, int ldc, int ldd,
         void const *ptr_A, void const *ptr_B, void const *ptr_C, void *ptr_D,
         void const *alpha, void const *beta,
         ScalarPointerMode scalar_mode,
         const Operation *operation,
         triton::driver::stream *stream) {

  GemmConfiguration configuration{{M, N, K}, lda, ldb, ldc, ldd, 1};

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

  cudaStream_t cudaStream = (cudaStream_t)*stream->cu();
  // Initialize host and device workspaces
  Status status = operation->initialize(&configuration, host_workspace, device_workspace, cudaStream);
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error("Unable to initialize workspace");

  // Run the operator
  GemmArguments arguments{ptr_A, ptr_B, ptr_C, ptr_D, alpha, beta, scalar_mode};
  operation->run(&arguments, host_workspace, device_workspace, cudaStream);
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
                          triton::driver::stream *stream) {

  // index operation table with functional key
  GemmFunctionalKey key(
      Provider::kCUTLASS,
      GemmKind::kGemm,
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
  // assume 16-bytes aligned memory pointers
  int alignment = 16;
  GemmPreferenceKey preference_key(cc, alignment);
  auto autotune_it = operators_it->second.find(preference_key);
  if (autotune_it == operators_it->second.end())
    throw std::runtime_error("Unable to find gemm operation");
  const std::vector<const Operation *> &operations = autotune_it->second;
  if (operations.empty())
    throw std::runtime_error("Unable to find gemm operation");

  // check if configuration was already autotuned
  tune_key_t tune_key{key, preference_key, M, N, K};
  if (op_cache.find(tune_key) != op_cache.end())
    return op_cache[tune_key];

  // auto-tune
  const Operation *best = nullptr;
  double best_ms = std::numeric_limits<double>::max();
  for (const Operation *op : operations) {
    auto fn = [&]() { run(M, N, K, lda, ldb, ldc, ldd, ptr_A, ptr_B, ptr_C, ptr_D,
                          alpha, beta, scalar_mode, op, stream); };
    double ms = triton::tools::bench(fn, stream, 10, 25);
    if (ms < best_ms) {
      best_ms = ms;
      best = op;
    }
  }
  op_cache[tune_key] = best;
  return best;
}

// map of torch datatypes to cutlass datatypes
std::map<caffe2::TypeIdentifier, NumericTypeID> type_map = {
    {caffe2::TypeMeta::Id<at::Half>(), NumericTypeID::kF16},
    {caffe2::TypeMeta::Id<float>(), NumericTypeID::kF32},
    {caffe2::TypeMeta::Id<double>(), NumericTypeID::kF64}};

void cutlass_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  size_t M = A.size(0);
  size_t N = B.size(1);
  size_t K = A.size(1);
  size_t lda = A.stride(0);
  size_t ldb = B.stride(0);
  size_t ldc = C.stride(0);
  size_t ldd = C.stride(0);
  void *ptr_A = A.data_ptr<void>();
  void *ptr_B = B.data_ptr<void>();
  void *ptr_C = C.data_ptr<void>();
  void *ptr_D = ptr_C;
  float alpha = 1.0f;
  float beta = 0.0f;
  // layout for A
  LayoutTypeID layout_A;
  if (A.stride(0) == 1)
    layout_A = LayoutTypeID::kColumnMajor;
  else if (A.stride(1) == 1)
    layout_A = LayoutTypeID::kRowMajor;
  else {
    A = A.contiguous();
    layout_A = LayoutTypeID::kRowMajor;
  }
  // layout for B
  LayoutTypeID layout_B;
  if (B.stride(0) == 1)
    layout_B = LayoutTypeID::kColumnMajor;
  else if (B.stride(1) == 1)
    layout_B = LayoutTypeID::kRowMajor;
  else {
    B = B.contiguous();
    layout_B = LayoutTypeID::kRowMajor;
  }
  // data types
  NumericTypeID element_compute = NumericTypeID::kF32;
  NumericTypeID element_A = type_map[A.dtype().id()];
  NumericTypeID element_B = type_map[B.dtype().id()];
  NumericTypeID element_C = type_map[C.dtype().id()];
  // misc. flags
  ScalarPointerMode scalar_mode = ScalarPointerMode::kHost;
  NumericTypeID element_scalar = NumericTypeID::kF32;
  ComplexTransform transform_A = ComplexTransform::kNone;
  ComplexTransform transform_B = ComplexTransform::kNone;
}
