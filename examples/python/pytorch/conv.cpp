#include <torch/torch.h>
#include <torch/script.h>
#include "ATen/cuda/CUDAContext.h"
#include <vector>
#include "triton/jit.h"
#include "triton/driver/stream.h"

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const char* src =
R"(
const tunable int32 TM = {16, 32, 64};
const tunable int32 TN = {16, 32, 64};
const tunable int32 TK = {8};

__constant__ int32* delta = alloc_const int32[18];
__constant__ int32* masks = alloc_const int32[1024];

void conv(read_only restrict fp32 *a,
          read_only restrict fp32 *b,
          fp32 *c,
          int32 M, int32 N, int32 K,
          int32 AN, int32 AH, int32 AW,
          int32 CN, int32 CK, int32 CP, int32 CQ,
          int32 AC, int32 AR, int32 AS,
          int32 lda_n, int32 lda_c, int32 lda_h, int32 lda_w,
          int32 ldc_n, int32 ldc_k, int32 ldc_p, int32 ldc_q,
          int32 pad_h, int32 pad_w,
          int32 bound){
    int32 rxa[TM] = get_global_range[TM](0);
    int32 rb0[TN] = get_global_range[TN](1);
    int32 rka[TK] = 0 ... TK;
    int32 rb1[TK] = 0 ... TK;
    fp32 C[TM, TN] = 0;
    int32 ranh[TM] = rxa / CQ;
    int32 raw[TM] = rxa % CQ - pad_w;
    int32 ran[TM] = ranh / CP;
    int32 rah[TM] = ranh % CP - pad_h;
    int32 ra0[TM] = ran*lda_n + rah*lda_h + raw*lda_w;
    int32 racr[TK] = rka / AS;
    int32 ras[TK] = rka % AS;
    int32 rac[TK] = racr / AR;
    int32 rar[TK] = racr % AR;
    int32 ra1[TK] = rac*lda_c + rar*lda_h + ras*lda_w;
    fp32* pa[TM, TK] = a + ra1[newaxis, :] + ra0[:, newaxis];
    fp32* pb[TN, TK] = b + rb1[newaxis, :]*CK + rb0[:, newaxis];
    __constant__ int32* pincd[TK] = delta + rka;
    __constant__ int32* pd[TK] = delta + AR*AS + rka;
    int32 d[TK] = *pd;
    int32 incd[TK] = *pincd;
    int32 maskh[TM] = pad_h + min(rah, 0) + max(rah + AR - AH, 0);
    int32 maskw[TM] = pad_w + min(raw, 0) + max(raw + AS - AW, 0);
    __constant__ int32* pm[TM] = masks + AR*AS + maskw*AR*AS + maskh*AR*AS*(2*pad_w + 1);
    __constant__ int32* pincm[TM] = delta;
    int32 incm[TM] = *pincm;
    int32 checka0[TM] = *pm;
    int32 checka1[TK] = 1 << rka;
    int1 checka[TM, TK] = (checka0[:, newaxis] & checka1[newaxis, :]) > 0;
    fp32 a[TM, TK] = checka ? *pa : 0;
    fp32 b[TN, TK] = *pb;
    for(int32 k = K; k > 0; k = k - TK){
      C = dot(a, trans(b), C);
      pb = pb + TK*CK;
      pa = pa + d[newaxis, :];
      b = *pb;
      pd = pd + incd;
      pincd = pincd + incd;
      d = *pd;
      incd = *pincd;
      pm = pm + incm;
      pincm = pincm + incm;
      incm = *pincm;
      checka0 = *pm;
      checka = (checka0[:, newaxis] & checka1[newaxis, :]) > 0;
      a = checka ? *pa : 0;
    }
    int32 rxc[TM] = get_global_range[TM](0);
    int32 rc1[TN] = get_global_range[TN](1);
    int32 rcn[TM] = rxc / (CP*CQ);
    int32 rcpq[TM] = rxc % (CP*CQ);
    int32 rc0[TM] = rcn * ldc_n + rcpq;
    fp32* pc[TM, TN]  = c + rc1[newaxis, :]*ldc_k + rc0[:, newaxis];
    int1 checkc0[TM] = rxc < M;
    int1 checkc1[TN] = rc1 < N;
    int1 checkc[TM, TN]  = checkc0[:, newaxis] && checkc1[newaxis, :];
    @checkc *pc = C;
})";

void build_conv_lut(int TK,
                    int stride_d, int stride_h, int stride_w, int stride_c,
                    int pad_d, int pad_h, int pad_w,
                    int T, int R, int S,
                    std::vector<int>& res, std::vector<int>& masks) {
  /* convolution parameters */
  int F = T * R * S;
  int Nlut = (TK + F - 1) / F * F;
  int upsample_w = 1;
  int upsample_h = 1;
  int upsample_d = 1;
  /* unpack index wrt filters */
  auto unpack = [&](int32_t trs){
    int32_t tr = trs / S;
    int32_t s = trs - tr*S;
    int32_t t = tr / R;
    int32_t r = tr - t*R;
    return std::make_tuple(t, r, s);
  };
  /* increments */
  for(size_t i = 0; i < Nlut; ++i)
    res[i] = (((i + TK) % Nlut) - i);
  /* deltas */
  size_t Ds0 = Nlut;
  size_t Ds1 = upsample_w;
  size_t Ds2 = upsample_h;
  size_t Ds3 = upsample_d;
  for(size_t pd = 0; pd < Ds3; ++pd)
  for(size_t ph = 0; ph < Ds2; ++ph)
  for(size_t pw = 0; pw < Ds1; ++pw){
    int32_t* deltas_ptr = &res[Nlut + pw*Ds0 + ph*Ds0*Ds1 + pd*Ds0*Ds1*Ds2];
    // cumulative increments
    for(size_t i = 0; i < Ds0; ++i){
      int32_t ctrs = i;
      int32_t c = ctrs / F;
      int32_t t, r, s;
      std::tie(t, r, s) = unpack(ctrs % F);
      // next indices
      int32_t nextctrs = ctrs + TK;
      int32_t nextc = nextctrs / F;
      int32_t nextt, nextr, nexts;
      std::tie(nextt, nextr, nexts) = unpack(nextctrs % F);
      // diffs
      int32_t cdiff = nextc - c;
      int32_t tdiff = (nextt + pd)/upsample_d - (t + pd)/upsample_d;
      int32_t rdiff = (nextr + ph)/upsample_h - (r + ph)/upsample_h;
      int32_t sdiff = (nexts + pw)/upsample_w - (s + pw)/upsample_w;
      // delta pointers
      deltas_ptr[i] = cdiff*stride_c + sdiff*stride_w + rdiff*stride_h + tdiff*stride_d;
    }
  }

  /* Masks */
  size_t Ms0 = Nlut;
  size_t Ms1 = 2*pad_w + 1;
  size_t Ms2 = 2*pad_h + 1;
  size_t Ms3 = 2*pad_d + 1;

  for(size_t pd = 0; pd < Ms3; ++pd)
  for(size_t ph = 0; ph < Ms2; ++ph)
  for(size_t pw = 0; pw < Ms1; ++pw){
    int32_t* masks_ptr = &masks[Nlut + pw*Ms0 + ph*Ms0*Ms1 + pd*Ms0*Ms1*Ms2];
    for(size_t i = 0; i < Ms0; ++i){
       int32_t t, r, s;
       int32_t mask = 0x0;
       for(size_t j = 0; j < TK; ++j){
         std::tie(t, r, s) = unpack((i + j) % F);
         bool in_bounds_d = (t + pd) >= pad_d && (t + pd) < (T + pad_d);
         bool in_bounds_h = (r + ph) >= pad_h && (r + ph) < (R + pad_h);
         bool in_bounds_w = (s + pw) >= pad_w && (s + pw) < (S + pad_w);
         mask |= (in_bounds_d && in_bounds_h && in_bounds_w) << j;
       }
       masks_ptr[i] = mask;
    }
  }
  for(size_t i = 0; i < Nlut; ++i)
    masks[i] = 0x0;
}

torch::Tensor conv_forward(
    const torch::Tensor data,
    const torch::Tensor weight) {
  // Check
  CHECK_INPUT(data);
  CHECK_INPUT(weight);
  // Unpack data shapes
  const int32_t B  = data.size(0);
  const int32_t Ci = data.size(1);
  const int32_t H  = data.size(2);
  const int32_t W  = data.size(3);
  // Unpack weight shapes
  const int32_t Cf = weight.size(0);
  const int32_t T  = 1;
  const int32_t R  = weight.size(1);
  const int32_t S  = weight.size(2);
  const int32_t NF  = weight.size(3);
  // Conv parameters
  int32_t upsample_d = 1, upsample_h = 1, upsample_w = 1;
  int32_t pad_d = 0, pad_h = 0, pad_w = 0;
  int32_t stride_h = 1, stride_w = 1;
  // Output shapes
  int32_t P = (H*upsample_h - R + 1 + 2*pad_h + stride_h - 1)/stride_h;
  int32_t Q = (W*upsample_w - S + 1 + 2*pad_w + stride_w - 1)/stride_w;
  // Allocate output
  AT_CHECK(Ci == Cf, "Number of channels in data and weights must match");
  torch::Tensor output = torch::empty({B, NF, P, Q}, torch::kFloat).cuda();
  // Wrap CUDA handles
  c10::DeviceIndex device = output.storage().device().index();
  triton::driver::cu_stream sstream((CUstream)at::cuda::getCurrentCUDAStream(device).stream(), false);
  triton::driver::stream* stream = &sstream;
  triton::driver::context* ctx = stream->context();
  triton::driver::cu_buffer d(ctx, (CUdeviceptr)data.storage().data(), false);
  triton::driver::cu_buffer w(ctx, (CUdeviceptr)weight.storage().data(), false);
  triton::driver::cu_buffer a(ctx, (CUdeviceptr)output.storage().data(), false);
  // Create JIT
  triton::jit jit(ctx);
  std::vector<unsigned> params = {
    16, 2, 64,
    32, 2, 64,
    16, 8, 2, 2,
    8, 1, 8,
    4
  };
  jit.add_module("conv", src, params);
  triton::driver::kernel* kernel = jit.get_function("conv");
  triton::jit::launch_information info = jit.get_launch_info("conv");
  // launch info
  unsigned TM = info.global_range_size[0];
  unsigned TN = info.global_range_size[1];
  unsigned TK = jit.get_int("TK");
  // initialize constant memory
  int FS = T*R*S;
  int nlut = (TK + FS - 1) / FS * FS;
  std::vector<int> h_delta(nlut + upsample_d*upsample_h*upsample_w*nlut);
  std::vector<int> h_masks(nlut + (2*pad_h+1)*(2*pad_w+1)*(2*pad_d+1)*nlut);
  // memory stride for images
  int32_t stride_i_w = 1;
  int32_t stride_i_h = W*stride_i_w;
  int32_t stride_i_d = H*stride_i_h;
  int32_t stride_i_c = 1*stride_i_d;
  int32_t stride_i_n = Ci*stride_i_c;
  // memory stride for activations
  int32_t stride_o_q = 1;
  int32_t stride_o_p = Q*stride_o_q;
  int32_t stride_o_m = P*stride_o_p;
  int32_t stride_o_k = 1*stride_o_m;
  int32_t stride_o_n = NF*stride_o_k;
  build_conv_lut(TK, stride_i_d, stride_i_h, stride_i_w, stride_i_c, pad_d, pad_h, pad_w, T, R, S, h_delta, h_masks);
  // equivalent matmul dimensions
  int32_t M = B*P*Q;
  int32_t N = NF;
  int32_t K = Ci*R*S;
  triton::driver::buffer* delta = jit.get_buffer("delta");
  triton::driver::buffer* masks = jit.get_buffer("masks");
  stream->write(delta, false, 0, h_delta.size()*4, h_delta.data());
  stream->write(masks, false, 0, h_masks.size()*4, h_masks.data());
  // launch info
  unsigned nthreads = info.num_threads;
  std::array<size_t, 3> grid = {(M + TM - 1)/TM, (N + TN - 1)/TN, 1};
  // fast bounds-checking
  unsigned lasti = (grid[0]*TM - 1)*TM + TM - 1;
  unsigned lastj = (grid[1]*TN - 1)*TN + TN - 1;
  unsigned lastk = TK - 1;
  bool AT = false;
  bool BT = true;
  unsigned last_safe_a = (AT==false)?(M*K - 1 - lasti)/M - lastk : M*K - 1 - lasti*K - lastk;
  unsigned last_safe_b =  (BT==true)?(N*K - 1 - lastj)/N - lastk : N*K - 1 - lastj*K - lastk;
  int32_t bound = std::max<unsigned>(1, std::max(K - last_safe_a, K - last_safe_b));
  // set arguments
  kernel->setArg(0, *d.cu());
  kernel->setArg(1, *w.cu());
  kernel->setArg(2, *a.cu());
  kernel->setArg(3, M);
  kernel->setArg(4, N);
  kernel->setArg(5, K);
  kernel->setArg(6, B);
  kernel->setArg(7, H);
  kernel->setArg(8, W);
  kernel->setArg(9, NF);
  kernel->setArg(10, P);
  kernel->setArg(11, Q);
  kernel->setArg(12, Ci);
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
  kernel->setArg(25, bound);
//  // dry run
  stream->enqueue(kernel, grid, {nthreads, 1, 1});
  return output;
}

static auto registry =
  torch::jit::RegisterOperators("triton::conv_forward", &conv_forward);
