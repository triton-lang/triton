#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor shift_cuda_forward(
    const at::Tensor input,
    const at::Tensor shift);

at::Tensor shift_cuda_backward(
    const at::Tensor grad_input,
    const at::Tensor shift);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor shift_forward(
    const at::Tensor input,
    const at::Tensor shift) {
  CHECK_INPUT(input);
  CHECK_INPUT(shift);

  return shift_cuda_forward(input, shift);
}

at::Tensor shift_backward(
    const at::Tensor grad_input,
    const at::Tensor shift) {
  CHECK_INPUT(grad_input);
  CHECK_INPUT(shift);
  return shift_cuda_backward(grad_input, shift);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &shift_forward, "Shift forward (CUDA)");
  m.def("backward", &shift_backward, "Shift backward (CUDA)");
}
