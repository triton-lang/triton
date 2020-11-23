#include <iomanip>
#include <cstring>
#include <sstream>
#include <cstdio>
#include <tuple>
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/tools/bench.hpp"
#include "triton/external/half.hpp"
#include "triton/runtime/function.h"
#include "src/dot.h"
#include "cuda/cublas.h"
#include "util.h"


struct dot_arg_t{
  uintptr_t a;
  uintptr_t b;
  uintptr_t c;
  float alpha;
  int M;
  int N;
  int K;
  int lda;
  int ldb;
  int ldc;
  uintptr_t locks;
};

template<class T, bool AT, bool BT>
static void cc_dot(std::vector<T> &c, const std::vector<T> &a, const std::vector<T> &b,
                    size_t M, size_t N, size_t K){
  for(size_t m = 0; m < M; m++)
  for(size_t n = 0; n < N; n++){
    float acc = 0;
    for(size_t k = 0; k < K; k++)
      acc = acc + (!AT ? a[k*M + m] : a[m*K + k]) * (!BT ? b[n*K + k] : b[k*N + n]);
    c[m*N + n] = static_cast<T>(acc);
  }
}

template<class T>
void cc_dot(bool AT_, bool BT_, size_t M, size_t N, size_t K,
             std::vector<T> &c, const std::vector<T> &a, const std::vector<T> &b) {
  if(AT_ && BT_)
    cc_dot<T, true, true>(c, a, b, M, N, K);
  else if(AT_ && !BT_)
    cc_dot<T, true, false>(c, a, b, M, N, K);
  else if(!AT_ && BT_)
    cc_dot<T, false, true>(c, a, b, M, N, K);
  else
    cc_dot<T, false, false>(c, a, b, M, N, K);
}

enum run_mode_t {
  BENCH,
  TEST
};

enum dtype_t {
  FLOAT,
  HALF,
  DOUBLE
};

template<class T>
struct to_string;

template<> struct to_string<half_float::half>{
  static constexpr const char* value = "half";
};

template<> struct to_string<float>{
  static constexpr const char* value = "float";
};

template<> struct to_string<double>{
  static constexpr const char* value = "double";
};

template<class T>
void triton_dot(drv::stream* stream, bool AT, bool BT,
                int32_t M, int32_t N, int32_t K,
                int32_t TM, int32_t TN, int32_t TK, int32_t nwarp,
                const std::vector<int>& a_order, const std::vector<int>& b_order,
                run_mode_t mode, std::vector<double>& bench, bool &test){
  std::string ty = to_string<T>::value;
  size_t dt_nbytes = sizeof(T);
  drv::context* context = stream->context();
  int32_t lda = (AT ^ a_order[0]==1) ? K : M;
  int32_t ldb = (BT ^ b_order[0]==1) ? N : K;
  int32_t ldc = N;
  std::vector<std::string> sa = { "1", "lda" };
  std::vector<std::string> sb = { "1", "ldb" };

  // inputs
  auto dc     = std::shared_ptr<drv::buffer>(drv::buffer::create(context, M*N*dt_nbytes));
  auto da     = std::shared_ptr<drv::buffer>(drv::buffer::create(context, M*K*dt_nbytes));
  auto db     = std::shared_ptr<drv::buffer>(drv::buffer::create(context, K*N*dt_nbytes));
  auto dlocks = std::shared_ptr<drv::buffer>(drv::buffer::create(context, 1024*1024*2*4));
//  ((drv::cu_buffer*)dlocks.get())->set_zero(stream, dlocks->size());

  // macros
  rt::function::options_space_t opt;
  // A access patterns
  opt.defines.push_back({"USEA",         {AT? "a"    : "a"            }});
  opt.defines.push_back({"BROADCAST_AK", {AT? "newaxis, :"   : "newaxis, :"   }});
  opt.defines.push_back({"BROADCAST_AM", {AT? ":, newaxis"   : ":, newaxis"   }});
  opt.defines.push_back({"SHAPE_A",      {AT? "TM, TK"       : "TM, TK"       }});
  opt.defines.push_back({"STRIDE_AK",    {AT? sa[a_order[0]] : sa[a_order[1]] }});
  opt.defines.push_back({"STRIDE_AM",    {AT? sa[a_order[1]] : sa[a_order[0]] }});
  // B access patterns
  opt.defines.push_back({"USEB",         {BT? "b"    : "b"            }});
  opt.defines.push_back({"BROADCAST_BK", {BT? ":, newaxis"   : ":, newaxis"   }});
  opt.defines.push_back({"BROADCAST_BN", {BT? "newaxis, :"   : "newaxis, :"   }});
  opt.defines.push_back({"SHAPE_B",      {BT? "TK, TN"       : "TK, TN"       }});
  opt.defines.push_back({"STRIDE_BK",    {BT? sb[b_order[1]] : sb[b_order[0]] }});
  opt.defines.push_back({"STRIDE_BN",    {BT? sb[b_order[0]] : sb[b_order[1]] }});
  // data-type
  opt.defines.push_back({"TYPE", {ty}});
  // tile sizes
  if(mode == TEST) {
    opt.defines.push_back({"TM", {std::to_string(TM)}});
    opt.defines.push_back({"TN", {std::to_string(TN)}});
    opt.defines.push_back({"TK", {std::to_string(TK)}});
    opt.defines.push_back({"TZ", {"1"}});
    opt.num_warps = {nwarp};
  }
  if(mode == BENCH) {
    opt.defines.push_back({"TM", {"128"}});
    opt.defines.push_back({"TN", {"128"}});
    opt.defines.push_back({"TK", {"32"}});
    opt.defines.push_back({"TZ", {"1"}});
    opt.num_warps = {4};
  }

  // kernels
  rt::function function(src::dot, opt);
  dot_arg_t args = {da->addr_as_uintptr_t(), db->addr_as_uintptr_t(), dc->addr_as_uintptr_t(),
                    1, M, N, K, lda, ldb, ldc, dlocks->addr_as_uintptr_t()};

  auto grid = [M, N](const rt::function::options_t& x) {
    return rt::grid_t{ceil(M, x.D<int>("TM"))*ceil(N, x.D<int>("TN")),
                      (size_t)1,
                      (size_t)x.D<int>("TZ")};
  };

  // metrics
  if(mode == BENCH){
    auto tflops = [&](double nanosec) { return 2.*M*N*K / nanosec * 1e-3; };
    double triton_ns = triton::tools::bench([&]() { function((void**)&args, sizeof(args), grid, stream);}, stream);
    bench.push_back(tflops(triton_ns));

    // cublas
   if(cublas::cublasinit()){
     T alpha(static_cast<double>(1));
     T beta(static_cast<double>(0));
     cublasGemmAlgo_t fastest;
//     cublasGemm(CUDA_R_16F, stream, AT, BT, M, N, K, &alpha, &*da, lda, &*db, ldb, &beta, &*dc, ldc, &fastest);
     double cublas_ms = triton::tools::bench([&]() { cublasGemm(CUDA_R_16F, stream, AT, BT, M, N, K,
                                                                &alpha, &*da, lda, &*db, ldb, &beta, &*dc,
                                                                ldc, nullptr, CUBLAS_GEMM_DEFAULT_TENSOR_OP); }, stream);
     bench.push_back(tflops(cublas_ms));
   }
  }

  // test triton
  if(mode == TEST){
    srand(0);
    // initialize buffers
    std::vector<T> hc(M*N);
    std::vector<T> ha(M*K);
    std::vector<T> hb(K*N);
    for(size_t i = 0; i < ha.size(); i++)
      ha[i] = (float)rand()/RAND_MAX;
    for(size_t i = 0; i < hb.size(); i++)
      hb[i] = (float)rand()/RAND_MAX;
    // copy buffer
    stream->write(&*da, true, 0, ha);
    stream->write(&*db, true, 0, hb);
    // run kernel
    function((void**)&args, sizeof(args), grid, stream);
    // write back
    stream->synchronize();
    // compare with CPU
    stream->read(&*dc, true, 0, hc);
    std::vector<T> rc(hc.size());
    cc_dot(AT, BT, M, N, K, rc, ha, hb);
    test = testing::diff(hc, rc);
  }
}

std::vector<double> bench_dot(drv::stream* stream,
               dtype_t dtype, bool AT, bool BT,
               int32_t M, int32_t N, int32_t K,
               const std::vector<int>& a_order, const std::vector<int>& b_order) {
  std::vector<double> bench;
  bool test;
  switch(dtype){
    case HALF:   triton_dot<half_float::half>(stream, AT, BT, M, N, K, 0, 0, 0, 0, a_order, b_order, BENCH, bench, test); break;
    case FLOAT:  triton_dot<float>(stream, AT, BT, M, N, K, 0, 0, 0, 0, a_order, b_order, BENCH, bench, test); break;
    case DOUBLE: triton_dot<double>(stream, AT, BT, M, N, K, 0, 0, 0, 0, a_order, b_order, BENCH, bench, test); break;
    default: break;
  }
  return bench;
}
bool test_dot(drv::stream* stream,
              dtype_t dtype, bool AT, bool BT,
              int32_t M, int32_t N, int32_t K,
              const std::vector<int>& a_order, const std::vector<int>& b_order,
              int32_t TM, int32_t TN, int32_t TK, size_t nwarp) {
  std::vector<double> bench;
  bool test = false;
  switch(dtype){
    case HALF:   triton_dot<half_float::half>(stream, AT, BT, M, N, K, TM, TN, TK, nwarp, a_order, b_order, TEST, bench, test); break;
    case FLOAT:  triton_dot<float>(stream, AT, BT, M, N, K, TM, TN, TK, nwarp, a_order, b_order, TEST, bench, test); break;
    case DOUBLE: triton_dot<double>(stream, AT, BT, M, N, K, TM, TN, TK, nwarp, a_order, b_order, TEST, bench, test); break;
    default: break;
  }
  return test;
}
