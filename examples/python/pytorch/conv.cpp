#include <torch/torch.h>
#include <torch/script.h>
#include "ATen/cuda/CUDAContext.h"
#include <vector>
#include "triton/jit.h"
#include "triton/driver/stream.h"
#include "triton/dnn/conv.h"

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor conv_common(
    int32_t B, int32_t C, int32_t D, int32_t H, int32_t W,
    int32_t T, int32_t R, int32_t S, int32_t NF,
    int32_t stride_d, int32_t stride_h, int32_t stride_w,
    int32_t pad_d, int32_t pad_h, int32_t pad_w,
    triton::dnn::conv::type ty,
    torch::Tensor torcha, torch::Tensor torchb
    ) {
  // Configuration
  triton::dnn::conv configuration(B, C, D, H, W, T, R, S, NF, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, ty);
  // Allocate output
  std::vector<int32_t> c_shapes = configuration.c_shapes();
  torch::Tensor torchc;
  if(ty == triton::dnn::conv::WGRAD)
    torchc = torch::empty({c_shapes[0], c_shapes[2], c_shapes[3], c_shapes[4]}, torch::kFloat).cuda();
  else
    torchc = torch::empty({c_shapes[0], c_shapes[1], c_shapes[3], c_shapes[4]}, torch::kFloat).cuda();
  // Wrap CUDA handles
  c10::DeviceIndex device = torchc.storage().device().index();
  triton::driver::cu_stream sstream((CUstream)at::cuda::getCurrentCUDAStream(device).stream(), false);
  triton::driver::stream* stream = &sstream;
  triton::driver::context* ctx = stream->context();
  triton::driver::cu_buffer a(ctx, (CUdeviceptr)torcha.storage().data(), false);
  triton::driver::cu_buffer b(ctx, (CUdeviceptr)torchb.storage().data(), false);
  triton::driver::cu_buffer c(ctx, (CUdeviceptr)torchc.storage().data(), false);
  stream->synchronize();
  // Create JIT
  triton::jit jit(ctx);
  std::string src = configuration.src();
  jit.add_module("conv", src.c_str(), configuration.default_params());
  triton::driver::kernel* kernel = jit.get_function("conv");
  triton::jit::launch_information info = jit.get_launch_info("conv");
  // launch info
  unsigned TM = info.global_range_size[0];
  unsigned TN = info.global_range_size[1];
  // launch info
  configuration.init(stream, jit);
  unsigned nthreads = info.num_threads;
  std::array<size_t, 3> grid = configuration.get_grid(TM, TN);
  configuration.set_arg(kernel, &a, &b, &c);
  stream->synchronize();
  stream->enqueue(kernel, grid, {nthreads, 1, 1});
  stream->synchronize();
  return torchc;
}

torch::Tensor conv_fprop(
    const torch::Tensor data,
    const torch::Tensor weight) {
  // Check
  CHECK_INPUT(data);
  CHECK_INPUT(weight);
  // Unpack data shapes
  const int32_t B  = data.size(0);
  const int32_t Ci = data.size(1);
  const int32_t D = 1;
  const int32_t H  = data.size(2);
  const int32_t W  = data.size(3);
  // Unpack weight shapes
  const int32_t Cf = weight.size(0);
  const int32_t T  = 1;
  const int32_t R  = weight.size(1);
  const int32_t S  = weight.size(2);
  const int32_t NF  = weight.size(3);
  // Configuration
  const int32_t stride_d = 1, stride_h = 1, stride_w = 1;
  const int32_t pad_d = 0, pad_h = 1, pad_w = 1;
  // Check
  AT_CHECK(Ci == Cf, "Number of channels in data and weights must match");
  return conv_common(B, Ci, D, H, W, T, R, S, NF, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, triton::dnn::conv::FPROP, data, weight);
}

torch::Tensor conv_bprop(
    const torch::Tensor derror,
    const torch::Tensor weight){
  // Check
  CHECK_INPUT(derror);
  CHECK_INPUT(weight);
  // Unpack data shapes
  const int32_t B  = derror.size(0);
  const int32_t Ki = derror.size(1);
  const int32_t M = 1;
  const int32_t P  = derror.size(2);
  const int32_t Q  = derror.size(3);
  // Unpack weight shapes
  const int32_t C = weight.size(0);
  const int32_t T  = 1;
  const int32_t R  = weight.size(1);
  const int32_t S  = weight.size(2);
  const int32_t Kw  = weight.size(3);
  // Compute M, P, Q
  const int32_t upsample_d = 1, upsample_h = 1, upsample_w = 1;
  const int32_t stride_d = 1, stride_h = 1, stride_w = 1;
  const int32_t pad_d = 0, pad_h = 1, pad_w = 1;
  const int32_t D = M*stride_d + T - 1 - 2*pad_d + stride_d - 1 / upsample_d;
  const int32_t H = P*stride_d + R - 1 - 2*pad_h + stride_h - 1 / upsample_h;
  const int32_t W = Q*stride_d + S - 1 - 2*pad_w + stride_w - 1 / upsample_w;
  // Check
  AT_CHECK(Ki == Kw, "Number of channels in error and weights must match");
  return conv_common(B, C, D, H, W, T, R, S, Kw, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, triton::dnn::conv::BPROP, derror, weight);
}

torch::Tensor conv_wgrad(
    const torch::Tensor data,
    const torch::Tensor derror
    ){
  // Check
  CHECK_INPUT(data);
  CHECK_INPUT(derror);
  // Unpack data shapes
  const int32_t Ba  = derror.size(0);
  const int32_t C = derror.size(1);
  const int32_t D = 1;
  const int32_t H  = derror.size(2);
  const int32_t W  = derror.size(3);
  // Unpack error shapes
  const int32_t Bb = derror.size(0);
  const int32_t K  = derror.size(1);
  const int32_t M  = 1;
  const int32_t P  = derror.size(2);
  const int32_t Q  = derror.size(3);
  // Compute M, P, Q
  const int32_t upsample_d = 1, upsample_h = 1, upsample_w = 1;
  const int32_t stride_d = 1, stride_h = 1, stride_w = 1;
  const int32_t pad_d = 0, pad_h = 1, pad_w = 1;
  const int32_t T = (D - M*stride_d + 1 + 2*pad_d - stride_d + 1)*upsample_d;
  const int32_t R = (H - P*stride_h + 1 + 2*pad_h - stride_h + 1)*upsample_h;
  const int32_t S = (W - Q*stride_w + 1 + 2*pad_w - stride_w + 1)*upsample_w;
  // Check
  AT_CHECK(Ba == Bb, "Number of channels in error and weights must match");
  return conv_common(Ba, C, D, H, W, T, R, S, K, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, triton::dnn::conv::WGRAD, data, derror);
}

static auto registry =
  torch::jit::RegisterOperators("triton::conv_fprop", &conv_fprop)
                            .op("triton::conv_bprop", &conv_bprop)
                            .op("triton::conv_wgrad", &conv_wgrad);
