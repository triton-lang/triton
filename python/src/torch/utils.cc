
#include "triton/driver/device.h"
#include "triton/driver/stream.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

namespace torch_utils {

uint64_t cu_device(int64_t dev_id) {
  CUdevice handle;
  triton::driver::dispatch::cuDeviceGet(&handle, dev_id);
  return (uint64_t)handle;
}

uint64_t cu_stream(int64_t dev_id) {
  return (uint64_t)c10::cuda::getCurrentCUDAStream(dev_id).stream();
}

void set_device(int64_t dev_id) {
  if (dev_id >= 0)
    C10_CUDA_CHECK(cudaSetDevice(dev_id));
}

} // namespace torch_utils

void init_torch_utils(pybind11::module &m) {
  pybind11::module subm = m.def_submodule("torch_utils");
  subm.def("cu_device", &torch_utils::cu_device);
  subm.def("cu_stream", &torch_utils::cu_stream);
  subm.def("set_device", &torch_utils::set_device);
}