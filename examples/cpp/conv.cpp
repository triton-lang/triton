#include <cstring>
#include <cstdio>
#include "common.hpp"
#include "triton/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/dnn/conv.h"

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  // initialize just-in-time compiler
  triton::jit jit(context);
  // initialization
  int32_t B = 4, NF = 32;
  int32_t D = 1, H = 24, W = 240;
  int32_t NC = 64, T = 1, R = 3, S = 3;
  int32_t pad_d = 0, pad_h = 1, pad_w = 1;
  int32_t stride_d = 1, stride_h = 1, stride_w = 1;
  int32_t upsample_d = 1, upsample_h = 1, upsample_w = 1;
  int32_t RD = (D*upsample_d - T + 1 + 2*pad_d + stride_d - 1)/stride_d;
  int32_t RH = (H*upsample_h - R + 1 + 2*pad_h + stride_h - 1)/stride_h;
  int32_t RW = (W*upsample_w - S + 1 + 2*pad_w + stride_w - 1)/stride_w;
  // equivalent matmul dimensions
  int32_t M = B*RD*RH*RW;
  int32_t N = NF;
  int32_t K = NC*T*R*S;
  std::vector<float> hc(B*RH*RW*NF);
  std::vector<float> rc(B*RH*RW*NF);
  std::vector<float> ha(B*NC*H*W);
  std::vector<float> hb(NC*R*S*NF);
  srand(0);
  for(size_t i = 0; i < ha.size(); i++)
    ha[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < hb.size(); i++)
    hb[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < hc.size(); i++)
    hc[i] = 0;
  triton::driver::buffer* dc = triton::driver::buffer::create(context, hc.size()*4);
  triton::driver::buffer* da = triton::driver::buffer::create(context, ha.size()*4);
  triton::driver::buffer* db = triton::driver::buffer::create(context, hb.size()*4);
  triton::driver::stream* stream = triton::driver::stream::create(context);
  stream->write(da, true, 0, ha);
  stream->write(db, true, 0, hb);
  stream->write(dc, true, 0, hc);
  stream->synchronize();
  // memory strides for data
  int32_t stride_i_w = 1;
  int32_t stride_i_h = W*stride_i_w;
  int32_t stride_i_d = H*stride_i_h;
  int32_t stride_i_c = D*stride_i_d;
  int32_t stride_i_n = NC*stride_i_c;
  // memory stride for activations
  int32_t stride_o_q = 1;
  int32_t stride_o_p = RW*stride_o_q;
  int32_t stride_o_m = RH*stride_o_p;
  int32_t stride_o_k = RD*stride_o_m;
  int32_t stride_o_n = NF*stride_o_k;
  // look-up table
  std::vector<int> h_delta, h_masks;
  triton::dnn::conv::init_cst(stride_i_d, stride_i_h, stride_i_w, stride_i_c, pad_d, pad_h, pad_w, T, R, S, h_delta, h_masks);
  // benchmark a given convolution kernel
  auto benchmark = [&](triton::driver::kernel* kernel,
                       triton::jit::launch_information info) {
    // launch info
    unsigned TM = info.global_range_size[0];
    unsigned TN = info.global_range_size[1];
    // initialize constant memory
    triton::driver::buffer* delta = jit.get_buffer("delta");
    triton::driver::buffer* masks = jit.get_buffer("masks");
    stream->write(delta, false, 0, h_delta.size()*4, h_delta.data());
    stream->write(masks, false, 0, h_masks.size()*4, h_masks.data());
    stream->synchronize();
    // launch info
    unsigned nthreads = info.num_threads;
    std::array<size_t, 3> grid = {(M + TM - 1)/TM, (N + TN - 1)/TN, 1};
    // set arguments
    kernel->setArg(0, da);
    kernel->setArg(1, db);
    kernel->setArg(2, dc);
    kernel->setArg(3, M);
    kernel->setArg(4, N);
    kernel->setArg(5, K);
    kernel->setArg(6, B);
    kernel->setArg(7, H);
    kernel->setArg(8, W);
    kernel->setArg(9, NF);
    kernel->setArg(10, RH);
    kernel->setArg(11, RW);
    kernel->setArg(12, NC);
    kernel->setArg(13, R);
    kernel->setArg(14, S);
    kernel->setArg(15, stride_i_n);
    kernel->setArg(16, stride_i_c);
    kernel->setArg(17, stride_i_h);
    kernel->setArg(18, stride_i_w);
    kernel->setArg(19, stride_o_n);
    kernel->setArg(20, stride_o_k);
    kernel->setArg(21, stride_o_p);
    kernel->setArg(22, stride_o_q);
    kernel->setArg(23, pad_h);
    kernel->setArg(24, pad_w);
    // dry run
    stream->enqueue(kernel, grid, {nthreads, 1, 1});
    stream->synchronize();
    // benchmark
    double ts = bench([&](){stream->enqueue(kernel, grid, {nthreads, 1, 1});},
                      [&](){ stream->synchronize(); }, *context->device());
    return 2.*M*N*K / ts * 1e-3;
  };
  std::string src = triton::dnn::conv::src();
//  jit.autotune("conv", src.c_str(), benchmark);
  jit.add_module("conv", src.c_str(), triton::dnn::conv::default_params());
  triton::driver::kernel* kernel = jit.get_function("conv");
  triton::jit::launch_information info = jit.get_launch_info("conv");
  std::cout << "Performance: " << benchmark(kernel, info) << " TFLOPS " << std::endl;
  stream->read(dc, true, 0, hc);
  cpp_conv_nchw(NC, B, NF, D, H, W, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, RD, RH, RW, rc, ha, hb);
  for(size_t i = 0; i < M*N; i++)
    if(std::abs(hc[i] - rc[i])/std::max(hc[i], rc[i]) > 1e-4){
      std::cout << i << " " << hc[i] << " " << rc[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  std::cout << "Pass!" << std::endl;
}
