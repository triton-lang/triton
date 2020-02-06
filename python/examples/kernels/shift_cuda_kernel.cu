#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__global__ void shift_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    const int32_t* __restrict__ shift,
    scalar_t* __restrict__ output,
    const int32_t B,
    const int32_t C,
    const int32_t H,
    const int32_t W) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t size = B*C*H*W;
  
  const int32_t CHW = C*H*W;
  const int32_t HW = H*W;
  const int32_t b = idx / CHW;
  const int32_t c = (idx - b*CHW) / HW;
  const int32_t h = (idx - b*CHW - c*HW) / W;
  const int32_t w = idx - b*CHW - c*HW - h*W;
  const int32_t target_w = w + shift[2*c];
  const int32_t target_h = h + shift[2*c + 1];
  const int32_t target_idx = b*CHW + c*HW + target_h*W + target_w;
  if (idx < size && target_w >= 0 && target_w < W && target_h >= 0 && target_h < H) {
      output[target_idx] = input[idx];
  }
}

template <typename scalar_t>
__global__ void shift_cuda_backward_kernel(
    const scalar_t* __restrict__ grad_input,
    scalar_t* __restrict__ grad_output,
    const int32_t* __restrict__ shift,
    const int32_t B,
    const int32_t C,
    const int32_t W,
    const int32_t H) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t size = B*C*W*H;
  const int32_t CWH = C*W*H;
  const int32_t WH = W*H;
  const int32_t b = idx / CWH;
  const int32_t c = (idx - b*CWH) / WH;
  const int32_t w = (idx - b*CWH - c*WH) / W;
  const int32_t h = idx - b*CWH - c*WH - w*H;
  const int32_t target_w = w - shift[2*c];
  const int32_t target_h = h - shift[2*c + 1];
  const int32_t target_idx = b*CWH + c*WH + target_w*W + target_h;
  if (idx < size && target_w >= 0 && target_w < W && target_h >= 0 && target_h < H) {
      grad_output[target_idx] = grad_input[idx];
  }
}
} // namespace

at::Tensor shift_cuda_forward(
    const at::Tensor input,
    const at::Tensor shift) {
  const auto B = input.size(0);
  const auto C = input.size(1);
  const auto H = input.size(2);
  const auto W = input.size(3);
  const auto size = B*C*W*H;
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  auto output = at::zeros_like(input);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "shift_forward_cuda", ([&] {
    shift_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.data<scalar_t>(),
        shift.data<int32_t>(),
        output.data<scalar_t>(),
        B,
        C,
        H,
        W);
  }));

  return output;
}

at::Tensor shift_cuda_backward(
    const at::Tensor grad_input,
    const at::Tensor shift) {
  const auto B = grad_input.size(0);
  const auto C = grad_input.size(1);
  const auto H = grad_input.size(2);
  const auto W = grad_input.size(3);
  const auto size = B*C*W*H;
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  auto grad_output = at::zeros_like(grad_input);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_input.type(), "shift_backward_cuda", ([&] {
    shift_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_input.data<scalar_t>(),
        grad_output.data<scalar_t>(),
        shift.data<int32_t>(),
        B,
        C,
        H,
        W);
  }));

  return grad_output;
}
