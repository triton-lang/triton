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

torch::Tensor conv_forward(
    const torch::Tensor data,
    const torch::Tensor weight) {
  // Check
  CHECK_INPUT(data);
  CHECK_INPUT(weight);
  // Unpack data shapes
  const auto B  = data.size(0);
  const auto Ci = data.size(1);
  const auto H  = data.size(2);
  const auto W  = data.size(3);
  // Unpack weight shapes
  const auto Cf = weight.size(0);
  const auto R  = weight.size(1);
  const auto S  = weight.size(2);
  const auto K  = weight.size(3);
  // Allocate output
  AT_CHECK(Ci == Cf, "Number of channels in data and weights must match");
  torch::Tensor output = torch::empty({B, K, H, W}, torch::kFloat);
  // Wrap CUDA handles
  triton::driver::cu_stream sstream(at::cuda::getCurrentCUDAStream(), false);
  triton::driver::stream* stream = &sstream;
  triton::driver::context* ctx = stream->context();
  triton::driver::cu_buffer d(ctx, (CUdeviceptr)data.storage().data(), false);
  triton::driver::cu_buffer w(ctx, (CUdeviceptr)weight.storage().data(), false);
  // Create JIT
  triton::jit jit(ctx);
  std::vector<unsigned> params = {
    16, 2, 64,
    32, 2, 64,
    16, 8, 2, 2,
    8, 8,
    4
  };
  jit.add_module("conv", src, params);
  triton::driver::kernel* kernel = jit.get_function("conv");
  triton::jit::launch_information info = jit.get_launch_info("conv");

  return output;
}

static auto registry =
  torch::jit::RegisterOperators("triton::conv_forward", &conv_forward);
