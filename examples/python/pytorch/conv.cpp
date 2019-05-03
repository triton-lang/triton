#include <torch/torch.h>
#include <vector>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor conv_forward(
    const at::Tensor data,
    const at::Tensor weight) {
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
  // Create output
  AT_CHECK(Ci == Cf, "Number of channels in data and weights must match");
  return at::empty({B, K, H, W}, at::kFloat);
}

static auto registry =
  torch::jit::RegisterOperators("triton::conv::forward", &conv_forward);
