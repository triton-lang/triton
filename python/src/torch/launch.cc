// Thanks to Scott Gray (OpenAI) for the idea to pass the arguments
// as a string constructed with struct.pack in python

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "triton/driver/buffer.h"
#include "triton/driver/stream.h"
#include "triton/runtime/function.h"
#include "triton/tools/bench.hpp"
#include "torch/script.h"
#include "ATen/cuda/CUDAContext.h"
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>


namespace rt = triton::runtime;
namespace drv = triton::driver;

typedef std::pair<int, int> map_key_t;
extern std::map<map_key_t, std::shared_ptr<rt::function::grid_fn_ty>> id_grid_map;
extern std::map<int, std::shared_ptr<rt::function>> id_fn_map;
extern std::map<int, std::shared_ptr<drv::device>> tt_devices;
extern std::map<int, std::shared_ptr<drv::stream>> tt_streams;


int64_t cdiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

int64_t largest_pow2_divisor(int64_t a){
  if(a % 8 == 0) return 8;
  if(a % 4 == 0) return 4;
  if(a % 2 == 0) return 2;
  return 1;
}

int64_t cdiv_sum(torch::Tensor x, int64_t div){
  TORCH_CHECK(!x.is_cuda(), "Argument of cdiv_sum must be a CPU tensor")
  auto _x  = x.accessor<int, 1>();
  int64_t ret = 0;
  for(size_t i = 0; i < x.size(0); i++)
    ret += (_x[i] + div - 1) / div;
  return ret;
}

CUstream torch_get_cuda_stream(int64_t dev_id) {
  return (CUstream)c10::cuda::getCurrentCUDAStream(dev_id).stream();
}

CUdeviceptr torch_get_cuda_device(int64_t dev_id) {
  CUdevice ret;
  triton::driver::dispatch::cuDeviceGet(&ret, dev_id);
  return ret;
}

void synchronize(int64_t dev_id) {
  tt_streams[dev_id]->synchronize();
}

torch::Tensor cuda_empty_like(torch::Tensor x){
  if(x.nbytes() == 0)
    return torch::empty_like(x);
  void* data;
  cudaMalloc(&data, x.nbytes());
  auto ret = torch::from_blob((void*)data, x.sizes(), x.strides(), [data](void* ptr) { cudaFree(data); }, x.options());
  return ret;
}

void cuda_set_device(int64_t dev_id) {
  if(dev_id >= 0)
    C10_CUDA_CHECK(cudaSetDevice(dev_id));
}


void init_launch(pybind11::module &m) {
  m.def("cuda_set_device", &cuda_set_device);
  m.def("cuda_empty_like", &cuda_empty_like);
  m.def("largest_pow2_divisor", &largest_pow2_divisor);
  m.def("cdiv", &cdiv);
  m.def("cdiv_sum", &cdiv_sum);
  m.def("synchronize", &synchronize);
}