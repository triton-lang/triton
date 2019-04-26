#include <cstring>
#include <cstdio>
#include "common.hpp"
#include "triton/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"

// K = channels
// M = batch * height * width
// N = number of feature maps

const char* src =
R"(
const tunable int32 TM = {16, 32, 64, 128};
const tunable int32 TN = {16, 32, 64, 128};
const tunable int32 TK = {8};

__constant__ int32* delta = alloc_const int32[256];
__constant__ int32* masks = alloc_const int32[8192];

void shift(restrict read_only fp32 *a, restrict read_only fp32 *b, fp32 *c,
           int32 M, int32 N, int32 K,
           int32 ABS, int32 AH, int32 AW, int32 AR, int32 AS){
  int32 rxa[TM] = get_global_range[TM](0);
  int32 ryb[TN] = get_global_range[TN](1);
  int32 rka[TK] = 0 ... TK;
  int32 rkb[TK] = 0 ... TK;
  fp32 C[TM, TN] = 0;
  fp32* pxa[TM, TK] = a + rxa[:, newaxis];
  fp32* pb[TN, TK] = b + rkb[newaxis, :]*N + ryb[:, newaxis];
  __constant__ int32* pd[TK] = delta + rka;
  int32 pad_h = AR/2;
  int32 pad_w = AS/2;
  int32 rawhc[TM] = rxa / ABS;
  int32 raw[TM] = rawhc % AW - pad_w;
  int32 rahc[TM] = rawhc / AW;
  int32 rah[TM] = rahc % AH - pad_h;
  int32 maskh[TM] = pad_h + min(rah, 0) + max(rah + AR - AH, 0);
  int32 maskw[TM] = pad_w + min(raw, 0) + max(raw + AS - AW, 0);
  __constant__ int32* pxm[TM] = masks + maskh*K + maskw*K*(2*pad_h + 1);
  __constant__ int32* pm[TM, TK] = pxm[:, newaxis] + rka[newaxis, :];
  for(int32 k = K; k > 0; k = k - TK){
    int32 delta[TK] = *pd;
    fp32 *pa[TM, TK] = pxa + delta[newaxis, :];
    int1 m[TM, TK] = *pm > 0;
    fp32 a[TM, TK] = m ? *pa : 0;
    fp32 b[TN, TK] = *pb;
    C = dot(a, trans(b), C);
    pb = pb + TK*N;
    pd = pd + TK;
    pm = pm + TK;
  }
  int32 rxc[TM] = get_global_range[TM](0);
  int32 ryc[TN] = get_global_range[TN](1);
  fp32* pc[TM, TN] = c + ryc[newaxis, :]*M + rxc[:, newaxis];
  int1 checkc0[TM] = rxc < M;
  int1 checkc1[TN] = ryc < N;
  int1 checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  @checkc *pc = C;
}
)";

std::vector<int32_t> shift_deltas(// strides
                                  int32_t stride_w, int32_t stride_h, int32_t stride_c,
                                  // shift
                                  int32_t C,
                                  const std::vector<int32_t>& shift_h,
                                  const std::vector<int32_t>& shift_w) {
  std::vector<int32_t> res(C);
  for(unsigned c = 0; c < C; c++){
    res[c] = c*stride_c;
    res[c] += shift_h[c]*stride_h;
    res[c] += shift_w[c]*stride_w;
  }
  return res;
}

std::vector<int32_t> shift_masks(int32_t C,
                                 const std::vector<int32_t>& shift_h,
                                 const std::vector<int32_t>& shift_w,
                                 int32_t R, int32_t S) {
  size_t S0 = C;
  size_t S1 = R;
  size_t S2 = S;
  std::vector<int32_t> res(S0*S1*S2);
  for(size_t ph = 0; ph < S1; ++ph)
  for(size_t pw = 0; pw < S2; ++pw){
    int32_t* ptr = &res[ph*S0 + pw*S0*S1];
    for(size_t i = 0; i < S0; ++i){
      bool in_bounds_h = shift_h[i] + ph >= 0 && shift_h[i] + ph < R;
      bool in_bounds_w = shift_w[i] + pw >= 0 && shift_w[i] + pw < S;
      ptr[i] = in_bounds_h && in_bounds_w;
    }
  }
  return res;
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
  // equivalent matmul dimensions
  int32_t M = BS*H*W;
  int32_t N = F;
  int32_t K = C;
  std::cout << M << " " << N << " " << K << std::endl;
  std::vector<float> hc(BS*H*W*F);
  std::vector<float> rc(BS*H*W*F);
  std::vector<float> ha(BS*C*H*W);
  std::vector<float> hb(F*C);
  // strides
  int32_t stride_i_bs = 1;
  int32_t stride_i_w = BS*stride_i_bs;
  int32_t stride_i_h = W*stride_i_w;
  int32_t stride_i_c = H*stride_i_h;
  // random shifts
  std::vector<int32_t> shift_h(C);
  std::vector<int32_t> shift_w(C);
  for(int32_t c = 0; c < C; c++){
    shift_h[c] = rand() % R - R/2;
    shift_w[c] = rand() % S - S/2;
  }
  // initialize buffers
  srand(0);
  for(int c = 0 ; c < C; c++)
  for(int h = 0 ; h < H; h++)
  for(int w = 0 ; w < W; w++)
  for(int bs = 0 ; bs < BS; bs++){
    float value = (float)rand() / RAND_MAX;
    size_t idx = bs + w*stride_i_w + h*stride_i_h + c*stride_i_c;
    ha[idx] = value;
  }
  for(size_t i = 0; i < hb.size(); i++)
    hb[i] = (float)rand() / RAND_MAX;
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
  std::vector<int32_t> h_delta = shift_deltas(stride_i_w, stride_i_h, stride_i_c, C, shift_h, shift_w);
  std::vector<int32_t> h_masks = shift_masks(C, shift_h, shift_w, R, S);
  // benchmark a given matrix multiplication kernel
  auto benchmark = [&](triton::driver::kernel* kernel,
                       triton::jit::launch_information info) {
    // launch info
    unsigned TM = info.global_range_size[0];
    unsigned TN = info.global_range_size[1];
    unsigned nthreads = info.num_threads;
    // initialize constant memory
    triton::driver::buffer* delta = jit.get_buffer("delta");
    triton::driver::buffer* masks = jit.get_buffer("masks");
    stream->write(delta, false, 0, h_delta.size()*4, h_delta.data());
    stream->write(masks, false, 0, h_masks.size()*4, h_masks.data());
    stream->synchronize();
    // set argument
    kernel->setArg(0, da);
    kernel->setArg(1, db);
    kernel->setArg(2, dc);
    kernel->setArg(3, M);
    kernel->setArg(4, N);
    kernel->setArg(5, K);
    kernel->setArg(6, BS);
    kernel->setArg(7, H);
    kernel->setArg(8, W);
    kernel->setArg(9, R);
    kernel->setArg(10, S);
    // dry run
    std::array<size_t, 3> grid = {(M + TM - 1)/TM, (N + TN - 1)/TN, 1};
    stream->enqueue(kernel, grid, {nthreads, 1, 1});
    stream->synchronize();
    // benchmark
    double ts = bench([&](){stream->enqueue(kernel, grid, {nthreads, 1, 1});},
                      [&](){ stream->synchronize(); }, *context->device());
    ts = ts * 1e-9;
    double tflops = 2.*M*N*K / ts * 1e-12;
    return tflops;
  };

  // shift
  std::vector<unsigned> params = {
    16, 2, 64,
    32, 2, 64,
    16, 8, 2, 2,
    8, 8,
    4
  };
  jit.autotune("shift", src, benchmark);
  jit.add_module("shift", src, params);
  triton::driver::kernel* kernel = jit.get_function("shift");
  triton::jit::launch_information info = jit.get_launch_info("shift");
  std::cout << "Performance: " << benchmark(kernel, info) << " TFLOPS " << std::endl;
  stream->read(dc, true, 0, hc);
  shift_conv(C, H, W, BS, F, rc, ha, hb, shift_h, shift_w);
  for(size_t i = 0; i < M*N; i++)
    if(std::abs(hc[i] - rc[i])/std::max(hc[i], rc[i]) > 1e-4){
      std::cout << i << " " << hc[i] << " " << rc[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  std::cout << "Pass!" << std::endl;

}
