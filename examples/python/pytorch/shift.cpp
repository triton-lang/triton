#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include "ATen/cuda/CUDAContext.h"
#include "triton/driver/stream.h"
#include "triton/dnn/shift.h"

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor shift_common(
    int32_t B, int32_t C, int32_t D, int32_t H, int32_t W,
    int32_t T, int32_t R, int32_t S, int32_t F,
    int32_t stride_h, int32_t stride_w,
    int32_t* shift_h, int32_t* shift_w,
    triton::dnn::shift::type ty,
    torch::Tensor torcha, torch::Tensor torchb, torch::Tensor torchbias,
    bool autotune = false
    ) {
  // Wrap CUDA handles
  c10::DeviceIndex device = torcha.storage().device().index();
  CUstream custream = (CUstream)at::cuda::getCurrentCUDAStream(device).stream();
  triton::driver::cu_stream stream(custream, false);
  triton::driver::context* ctx = stream.context();
  // Get configuration
  bool has_bias = torchbias.storage().size() > 0;
  triton::dnn::shift shift(B, C, D, H, W, T, R, S, F,
                           stride_h, stride_w,
                           shift_h, shift_w, "fp32", "fp32",
                           ty, has_bias);
  // Bind memory
  triton::driver::cu_buffer a(ctx, (CUdeviceptr)torcha.storage().data(), false);
  triton::driver::cu_buffer b(ctx, (CUdeviceptr)torchb.storage().data(), false);
  triton::driver::cu_buffer cubias(ctx, (CUdeviceptr)torchbias.storage().data(), false);
  triton::driver::buffer* bias = has_bias ? &cubias : nullptr;
  // Allocate output
  std::vector<int32_t> _c_shapes = shift.c_shapes();
  std::vector<long int> c_shapes;
  for(auto x: _c_shapes)
    c_shapes.push_back(x);
  torch::Tensor torchc = torch::empty(c_shapes).cuda();
  triton::driver::cu_buffer c(ctx, (CUdeviceptr)torchc.storage().data(), false);
  // Enqueue
  shift.enqueue(&stream, {&a, &b, &c});
  return torchc;
}

torch::Tensor shift_y(
    const torch::Tensor x,
    const torch::Tensor w,
    const torch::Tensor bias,
    int64_t R, int64_t S,
    int64_t stride_h, int64_t stride_w,
    const torch::Tensor shift_h, const torch::Tensor shift_w) {
  CHECK_INPUT(x);
  CHECK_INPUT(w);
  // shapes for a
  int64_t Ca  = x.size(0);
  int64_t H   = x.size(1);
  int64_t W   = x.size(2);
  int64_t B   = x.size(3);
  // shapes for b
  int64_t Cb  = w.size(0);
  int64_t F   = w.size(1);
  AT_CHECK(Ca == Cb, "operands must have the same number of channels");
  int64_t C = Ca;
  // run
  return shift_common(B, C, 1, H, W, 1, R, S, F, stride_h, stride_w,
                     (int32_t*)shift_h.storage().data(), (int32_t*)shift_w.storage().data(),
                     triton::dnn::shift::FPROP, x, w, bias);
}

torch::Tensor shift_dx(
    const torch::Tensor dy,
    const torch::Tensor w,
    const torch::Tensor bias,
    int64_t R, int64_t S,
    int64_t stride_h, int64_t stride_w,
    const torch::Tensor shift_h, const torch::Tensor shift_w) {
  CHECK_INPUT(dy);
  CHECK_INPUT(w);
  // shapes for a
  int64_t Ca  = dy.size(0);
  int64_t H   = dy.size(1);
  int64_t W   = dy.size(2);
  int64_t B   = dy.size(3);
  H *= stride_h;
  W *= stride_w;
  // shapes for b
  int64_t Cb  = w.size(0);
  int64_t F   = w.size(1);
  std::swap(Cb, F);
  // checks
  AT_CHECK(Ca == Cb, "operands must have the same number of channels");
  int64_t C = Ca;
  std::swap(C, F);
  // run
  return shift_common(B, C, 1, H, W, 1, R, S, F, stride_h, stride_w,
                     (int32_t*)shift_h.storage().data(), (int32_t*)shift_w.storage().data(),
                     triton::dnn::shift::BPROP, dy, w, bias);
}

torch::Tensor shift_dw(
    const torch::Tensor dy,
    const torch::Tensor x,
    const torch::Tensor bias,
    int64_t R, int64_t S,
    int64_t stride_h, int64_t stride_w,
    const torch::Tensor shift_h, const torch::Tensor shift_w) {
  CHECK_INPUT(dy);
  CHECK_INPUT(x);
  // shapes for a
  int64_t F    = dy.size(0);
  int64_t Ha   = dy.size(1);
  int64_t Wa   = dy.size(2);
  int64_t Ba   = dy.size(3);
  // shapes for b
  int64_t C     = x.size(0);
  int64_t Hb    = x.size(1);
  int64_t Wb    = x.size(2);
  int64_t Bb    = x.size(3);
  // check
  AT_CHECK(Ha*stride_h == Hb, "operands must have the same image height");
  AT_CHECK(Wa*stride_w == Wb, "operands must have the same image width");
  AT_CHECK(Ba == Bb,          "operands must have the same batch size");
  int64_t H = Hb;
  int64_t W = Wb;
  int64_t B = Bb;
  // run
  return shift_common(B, C, 1, H, W, 1, R, S, F, stride_h, stride_w,
                     (int32_t*)shift_h.storage().data(), (int32_t*)shift_w.storage().data(),
                     triton::dnn::shift::WGRAD, dy, x, bias);
}

static auto registry =
  torch::jit::RegisterOperators("triton::shift_conv_y", &shift_y)
                            .op("triton::shift_conv_dx", &shift_dx)
                            .op("triton::shift_conv_dw", &shift_dw);
