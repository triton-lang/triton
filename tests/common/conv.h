#include <iomanip>
#include <cstring>
#include <sstream>
#include <cstdio>
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/tools/bench.hpp"
#include "triton/external/half.hpp"
#include "triton/runtime/function.h"
#include "src/conv.h"
#include "cuda/cublas.h"
#include "util.h"


struct conv_arg_t{
  CUdeviceptr a;
  CUdeviceptr b;
  CUdeviceptr c;
  float alpha;
  int M;
  int N;
  int K;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  CUdeviceptr adelta;
  int lda_z;
  int lda_ci;
  int lda_h;
  int lda_w;
  int ldb_ci;
  int ldb_r;
  int ldb_s;
  int ldb_co;
  int ldc_z;
  int ldc_co;
  int ldc_p;
  int ldc_q;
};

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
void triton_conv(drv::context* context, drv::stream* stream,
                int Z, int CI, int H, int W, int CO, int R, int S,
                int pad_h, int pad_w, int stride_h, int stride_w,
                run_mode_t mode, std::vector<double>& bench, bool &test){
  std::string ty = to_string<T>::value;
  size_t dt_nbytes = sizeof(T);
  drv::device* device = context->device();

  int P = (H + 2*pad_h - R)/stride_h + 1;
  int Q = (W + 2*pad_w - S)/stride_w + 1;

  // inputs
  auto dc     = std::shared_ptr<drv::buffer>(drv::buffer::create(context, Z*CO*P*Q*dt_nbytes));
  auto da     = std::shared_ptr<drv::buffer>(drv::buffer::create(context, Z*CI*H*W*dt_nbytes));
  auto db     = std::shared_ptr<drv::buffer>(drv::buffer::create(context, CI*R*S*CO*dt_nbytes));
  auto ddelta = std::shared_ptr<drv::buffer>(drv::buffer::create(context, CI*R*S*4));
  auto dlocks = std::shared_ptr<drv::buffer>(drv::buffer::create(context, 1024*1024*2*4));
  ((drv::cu_buffer*)dlocks.get())->set_zero(stream, dlocks->size());

  std::vector<int32_t> hdelta(CI*R*S);
  int TK = 16;
  for(int i = 0; i < hdelta.size(); i++){
    int s = i % S;
    int cr = i / S;
    int r = cr % R;
    int c = cr / R;
    int nexti = i + TK;
    int nexts = nexti % S;
    int nextcr = nexti / S;
    int nextr = nextcr % R;
    int nextc = nextcr / R;
    hdelta[i] = (nextc - c)*W*H + (nextr - r)*W + (nexts - s);
  }
  stream->write(&*ddelta, true, 0, hdelta);

  // macros
  rt::function::options_space_t opt;
  opt.defines.push_back({"TYPE", {ty}});
  opt.defines.push_back({"TM", {"128"}});
  opt.defines.push_back({"TN", {"128"}});
  opt.defines.push_back({"TK", {std::to_string(TK)}});
  opt.defines.push_back({"TZ", {"1"}});
  opt.defines.push_back({"RR", {std::to_string(R)}});
  opt.defines.push_back({"SS", {std::to_string(S)}});
  opt.defines.push_back({"PP", {std::to_string(P)}});
  opt.defines.push_back({"QQ", {std::to_string(Q)}});
  opt.defines.push_back({"HH", {std::to_string(H)}});
  opt.defines.push_back({"WW", {std::to_string(W)}});

  opt.num_warps = {2, 4};

  // kernels
  rt::function function(src::conv, opt);
  conv_arg_t args{*da->cu(), *db->cu(), *dc->cu(), 1, Z*P*Q, CO, CI*R*S,
                  pad_h, pad_w, stride_h, stride_w,
                  *ddelta->cu(),
                  W*H*CI, W*H, W, 1,
                  CO*S*R , CO*S, CO, 1,
                  Q*P*CO, Q*P, Q, 1};
  auto grid = [Z,P,Q,CO](const rt::function::options_t& x) {
    return rt::grid_t{ceil(Z*P*Q, x.D<int>("TM")),
                      ceil(CO   , x.D<int>("TN")),
                      (size_t)x.D<int>("TZ")};
  };
  auto tflops = [&](double nanosec) { return 2.*Z*P*Q*CI*CO*R*S / nanosec * 1e-3; };
  double triton_ns = triton::tools::bench([&]() { function((void**)&args, sizeof(args), grid, stream, device);}, stream);
  bench.push_back(tflops(triton_ns));
}

std::vector<double> bench_conv(drv::context* context, drv::stream* stream, dtype_t dtype,
               int32_t Z, int32_t H, int32_t W, int32_t CO, int32_t CI, int32_t R, int32_t S,
               int32_t pad_h, int32_t pad_w, int32_t stride_h, int32_t stride_w) {
  std::vector<double> bench;
  bool test;
  switch(dtype){
    case HALF:   triton_conv<half_float::half>(context, stream,  Z, CI, H, W, CO, R, S, pad_h, pad_w, stride_h, stride_w, BENCH, bench, test); break;
    case FLOAT:  triton_conv<float>(context, stream,  Z, CI, H, W, CO, R, S, pad_h, pad_w, stride_h, stride_w, BENCH, bench, test); break;
    case DOUBLE: triton_conv<double>(context, stream,  Z, CI, H, W, CO, R, S, pad_h, pad_w, stride_h, stride_w, BENCH, bench, test); break;
    default: break;
  }
  return bench;
}
