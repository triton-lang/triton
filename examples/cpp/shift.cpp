#include <cstring>
#include <cstdio>
#include <sstream>
#include "triton/runtime/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/tools/bench.hpp"
#include "triton/dnn/shift.h"

// input layout: C, H, W, BS
// filter layout: C, K
// output layout: K, H, W, BS
template<class IN_DTYPE, class OUT_DTYPE>
void shift_conv(int32_t C, int32_t H, int32_t W, int32_t BS,
                int32_t K,
                std::vector<OUT_DTYPE>& O,
                const std::vector<IN_DTYPE>& I,
                const std::vector<IN_DTYPE>& F,
                const std::vector<int32_t> shift_h,
                const std::vector<int32_t> shift_w)
{
  OUT_DTYPE acc;
  for(int32_t p = 0; p < H; ++p)
  for(int32_t q = 0; q < W; ++q)
  for(int32_t bs = 0; bs < BS; ++bs)
  for(int32_t k = 0; k < K; ++k)
  {
    acc = 0;
    for(int32_t c = 0; c < C; ++c){
      int32_t h = p + shift_h[c];
      int32_t w = q + shift_w[c];
      bool in_bounds = (h >= 0 && w >= 0 && h < H && w < W);
      IN_DTYPE a = in_bounds?I[bs + w*BS + h*BS*W + c*BS*H*W]:0;
      IN_DTYPE b = F[k + c*K];
      acc = std::fma(a, b, acc);
    }
    O[bs + q*BS + p*BS*W + k*BS*H*W] = acc;
  }
}

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  // initialize just-in-time compiler
  triton::jit jit(context);
  // initialization
  int32_t R = 3, S = 3;
  int32_t BS = 4, F = 128;
  int32_t H = 32, W = 32;
  int32_t C = 128;
  // random shifts
  std::vector<int32_t> shift_h(C);
  std::vector<int32_t> shift_w(C);
  for(int32_t c = 0; c < C; c++){
    shift_h[c] = rand() % R - R/2;
    shift_w[c] = rand() % S - S/2;
  }
  // configuration
  triton::dnn::shift shift(BS, C, 1, H, W, 1, R, S, F, shift_h, shift_w);
  // host buffers
  std::vector<float> hc(shift.c_size());
  std::vector<float> rc(shift.c_size());
  std::vector<float> ha(shift.a_size());
  std::vector<float> hb(shift.b_size());
  // device buffers
  triton::driver::buffer* dc = triton::driver::buffer::create(context, hc.size()*4);
  triton::driver::buffer* da = triton::driver::buffer::create(context, ha.size()*4);
  triton::driver::buffer* db = triton::driver::buffer::create(context, hb.size()*4);
  triton::driver::stream* stream = triton::driver::stream::create(context);
  // initialize host
  srand(0);
  for(size_t i = 0; i < ha.size(); i++)
    ha[i] = (float)rand() / RAND_MAX;
  for(size_t i = 0; i < hb.size(); i++)
    hb[i] = (float)rand() / RAND_MAX;
  for(size_t i = 0; i < hc.size(); i++)
    hc[i] = 0;
  // initialize device
  stream->write(da, true, 0, ha);
  stream->write(db, true, 0, hb);
  stream->write(dc, true, 0, hc);
  stream->synchronize();
  // benchmark
  auto benchmark = [&](triton::driver::kernel* kernel,
                       triton::jit::launch_information info) {
    shift.init(stream, (triton::driver::cu_module*)kernel->module());
    // launch info
    unsigned TM = info.global_range_size[0];
    unsigned TN = info.global_range_size[1];
    unsigned nthreads = info.num_threads;
    // set argument
    shift.enqueue(stream, kernel, da, db, dc, TM, TN, nthreads);
    stream->synchronize();
    // benchmark
    double ts = triton::tools::bench([&](){shift.enqueue(stream, kernel, da, db, dc, TM, TN, nthreads);},
                      [&](){ stream->synchronize(); }, context->device());
    return shift.get_nflops() / ts * 1e-3;
  };

  // shift
  std::vector<unsigned> params = {
    8, 2, 32, 8, 2, 64, 8, 4, 2, 2, 4, 2, 8, 4
  };
  std::ostringstream oss;
  shift.src(oss);
  std::string src = oss.str();
//  jit.autotune("shift", src.c_str(), benchmark);
  jit.add_module("shift", src.c_str(), params);
  triton::driver::kernel* kernel = jit.get_function("shift");
  triton::jit::launch_information info = jit.get_launch_info("shift");
  std::cout << "Performance: " << benchmark(kernel, info) << " TFLOPS " << std::endl;
  stream->read(dc, true, 0, hc);
  shift.cpu_ref(rc.data(), ha.data(), hb.data());
  for(size_t i = 0; i < hc.size(); i++)
    if(std::abs(hc[i] - rc[i])/std::max(hc[i], rc[i]) > 1e-4){
      std::cout << i << " " << hc[i] << " " << rc[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  std::cout << "Pass!" << std::endl;

}
