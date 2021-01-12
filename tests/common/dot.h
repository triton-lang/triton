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
void triton_dot(drv::context* context,  drv::stream* stream, bool AT, bool BT,
                int32_t M, int32_t N, int32_t K,
                int32_t TM, int32_t TN, int32_t TK, int32_t nwarp,
                const std::vector<int>& a_order, const std::vector<int>& b_order,
                run_mode_t mode, std::vector<double>& bench, bool &test){
  std::string ty = to_string<T>::value;
  size_t dt_nbytes = sizeof(T);
  drv::device* device = context->device();
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
  rt::options_space_t opts;
  // A access patterns
  opts.defines.push_back({"STRIDE_AK",    {AT? sa[a_order[0]] : sa[a_order[1]] }});
  opts.defines.push_back({"STRIDE_AM",    {AT? sa[a_order[1]] : sa[a_order[0]] }});
  // B access patterns
  opts.defines.push_back({"STRIDE_BK",    {BT? sb[b_order[1]] : sb[b_order[0]] }});
  opts.defines.push_back({"STRIDE_BN",    {BT? sb[b_order[0]] : sb[b_order[1]] }});
  // data-type
  opts.defines.push_back({"TYPE", {ty}});
  // tile sizes
  if(mode == TEST) {
    opts.defines.push_back({"TM", {std::to_string(TM)}});
    opts.defines.push_back({"TN", {std::to_string(TN)}});
    opts.defines.push_back({"TK", {std::to_string(TK)}});
    opts.defines.push_back({"TZ", {"1"}});
    opts.num_warps = {nwarp};
  }
  if(mode == BENCH) {
    opts.defines.push_back({"TM", {"128"}});
    opts.defines.push_back({"TN", {"128"}});
    opts.defines.push_back({"TK", {"32"}});
    opts.defines.push_back({"TZ", {"1"}});
    opts.num_warps = {4};
  }

  // arguments
  std::stringstream oss;
  rt::add_arg(oss, *da->cu());
  rt::add_arg(oss, *db->cu());
  rt::add_arg(oss, *dc->cu());
  rt::add_arg(oss, (float)1);
  rt::add_arg(oss, M);
  rt::add_arg(oss, N);
  rt::add_arg(oss, K);
  rt::add_arg(oss, lda);
  rt::add_arg(oss, ldb);
  rt::add_arg(oss, ldc);
  rt::add_arg(oss, *dlocks->cu());
  // kernel
  rt::function function(src::dot, opts);
  // grid
  auto grid = [M, N](const rt::options_t& x) {
    return rt::grid_t{ceil(M, x.D<int>("TM"))*
                      ceil(N, x.D<int>("TN")),
                      (size_t)x.D<int>("TZ")};
  };

  // metrics
  if(mode == BENCH){
    auto tflops = [&](double nanosec) { return 2.*M*N*K / nanosec * 1e-3; };
    double triton_ns = triton::tools::bench([&]() { function((void**)oss.str().data(), oss.str().size(), grid, stream, device);}, stream);
    bench.push_back(tflops(triton_ns));

    // cublas
   if(cublas::cublasinit()){
     T alpha(static_cast<double>(1));
     T beta(static_cast<double>(0));
     cublasGemmAlgo_t fastest;
//     cublasGemm(CUDA_R_16F, stream, AT, BT, M, N, K, &alpha, &*da, lda, &*db, ldb, &beta, &*dc, ldc, &fastest);
     double cublas_ms = triton::tools::bench([&]() { cublasGemm(CUDA_R_16F, stream, !AT, !BT, M, N, K,
                                                                &alpha, &*da, lda, &*db, ldb, &beta, &*dc,
                                                                ldc); }, stream);
     bench.push_back(tflops(cublas_ms));
   }
  }

//  rt::options_t opt;
//  for(auto &x: opts.defines)
//    opt.defines[x.first] = x.second[0];
//  opt.num_warps = 1;
//  std::cout << function.get_asm(rt::ASM_NV_PTX, device, opt) << std::endl;

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
    function((void**)oss.str().data(), oss.str().size(), grid, stream, device);
    // write back
    stream->synchronize();
    // compare with CPU
    stream->read(&*dc, true, 0, hc);
    std::vector<T> rc(hc.size());
    cc_dot(AT, BT, M, N, K, rc, ha, hb);
    test = testing::diff(hc, rc);
  }
}

std::vector<double> bench_dot(drv::context* context, drv::stream* stream,
               dtype_t dtype, bool AT, bool BT,
               int32_t M, int32_t N, int32_t K,
               const std::vector<int>& a_order, const std::vector<int>& b_order) {
  std::vector<double> bench;
  bool test;
  switch(dtype){
    case HALF:   triton_dot<half_float::half>(context, stream, AT, BT, M, N, K, 0, 0, 0, 0, a_order, b_order, BENCH, bench, test); break;
    case FLOAT:  triton_dot<float>(context, stream, AT, BT, M, N, K, 0, 0, 0, 0, a_order, b_order, BENCH, bench, test); break;
    case DOUBLE: triton_dot<double>(context, stream, AT, BT, M, N, K, 0, 0, 0, 0, a_order, b_order, BENCH, bench, test); break;
    default: break;
  }
  return bench;
}
bool test_dot(drv::context* context, drv::stream* stream,
              dtype_t dtype, bool AT, bool BT,
              int32_t M, int32_t N, int32_t K,
              const std::vector<int>& a_order, const std::vector<int>& b_order,
              int32_t TM, int32_t TN, int32_t TK, size_t nwarp) {
  std::vector<double> bench;
  bool test = false;
  switch(dtype){
    case HALF:   triton_dot<half_float::half>(context, stream, AT, BT, M, N, K, TM, TN, TK, nwarp, a_order, b_order, TEST, bench, test); break;
    case FLOAT:  triton_dot<float>(context, stream, AT, BT, M, N, K, TM, TN, TK, nwarp, a_order, b_order, TEST, bench, test); break;
    case DOUBLE: triton_dot<double>(context, stream, AT, BT, M, N, K, TM, TN, TK, nwarp, a_order, b_order, TEST, bench, test); break;
    default: break;
  }
  return test;
}
