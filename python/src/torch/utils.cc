
#include "triton/driver/device.h"
#include "triton/driver/stream.h"
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>

std::map<int, std::shared_ptr<triton::driver::device>> tt_devices;
std::map<int, std::shared_ptr<triton::driver::stream>> tt_streams;

namespace torch_utils {

void register_device(int64_t dev_id) {
  if (tt_devices.find(dev_id) != tt_devices.end())
    return;
  triton::driver::device *device;
  if (dev_id >= 0) {
    CUdevice handle;
    triton::driver::dispatch::cuDeviceGet(&handle, dev_id);
    device = new triton::driver::cu_device(handle, false);
  } else
    device = new triton::driver::host_device();
  tt_devices[dev_id].reset(device);
}

void register_stream(int64_t dev_id) {
  if (tt_streams.find(dev_id) != tt_streams.end())
    return;
  triton::driver::stream *stream;
  if (dev_id >= 0) {
    CUstream handle = (CUstream)c10::cuda::getCurrentCUDAStream(dev_id).stream();
    stream = new triton::driver::cu_stream(handle, false);
  } else
    stream = new triton::driver::host_stream();
  tt_streams[dev_id].reset(stream);
}

void synchronize(int64_t dev_id) {
  tt_streams[dev_id]->synchronize();
}

void set_device(int64_t dev_id) {
  if (dev_id >= 0)
    C10_CUDA_CHECK(cudaSetDevice(dev_id));
}

} // namespace torch_utils

void init_torch_utils(pybind11::module &m) {
  pybind11::module subm = m.def_submodule("torch_utils");
  subm.def("register_device", &torch_utils::register_device);
  subm.def("register_stream", &torch_utils::register_stream);
  subm.def("set_device", &torch_utils::set_device);
  subm.def("synchronize", &torch_utils::synchronize);
}