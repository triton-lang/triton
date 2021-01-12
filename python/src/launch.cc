// Thanks to Scott Gray (OpenAI) for the idea to pass the arguments
// as a string constructed with struct.pack in python

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
extern std::map<map_key_t, std::shared_ptr<rt::function>> id_fn_map;
std::shared_ptr<drv::device> host_device;
std::shared_ptr<drv::context> host_context;
std::shared_ptr<drv::stream> host_stream;

int64_t cdiv_sum(torch::Tensor x, int64_t div){
  TORCH_CHECK(!x.is_cuda(), "Argument of cdiv_sum must be a CPU tensor")
  auto _x  = x.accessor<int, 1>();
  int64_t ret = 0;
  for(size_t i = 0; i < x.size(0); i++)
    ret += (_x[i] + div - 1) / div;
  return ret;
}

void init_host_stream() {
  if(!host_stream){
    host_device.reset(new drv::host_device());
    host_context.reset(drv::context::create(&*host_device));
    host_stream.reset(drv::stream::create(host_context->backend()));
  }
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
  if(dev_id == -1){
    init_host_stream();
    host_stream->synchronize();
  }
  else{
    triton::driver::cu_stream stream(torch_get_cuda_stream(dev_id), false);
    stream.synchronize();
  }
}

torch::Tensor cuda_empty_like(torch::Tensor x){
  if(x.nbytes() == 0)
    return torch::empty_like(x);
  void* data;
  cudaMalloc(&data, x.nbytes());
  auto ret = torch::from_blob((void*)data, x.sizes(), x.strides(), [data](void* ptr) { cudaFree(data); }, x.options());
  return ret;
}

void launch_kernel(int64_t op_id, int64_t dev_id, const std::string& args, 
                   const std::vector<std::string>& constant_names, const std::vector<torch::Tensor>& constant_vals){
  rt::function* fn = id_fn_map.at({op_id, dev_id}).get();
  for(size_t n = 0; n < constant_names.size(); n++){
    const torch::Tensor& x = constant_vals[n];
    fn->set_cst(constant_names[n].c_str(), (char*)x.data_ptr(), x.numel()*x.element_size());
  }
  if(dev_id == -1){
    init_host_stream();
    (*fn)((void**)args.c_str(), args.size(), *id_grid_map.at({op_id, dev_id}), &*host_stream, &*host_device);
  }
  else{
    C10_CUDA_CHECK(cudaSetDevice(dev_id));
    triton::driver::cu_stream stream(torch_get_cuda_stream(dev_id), false);
    triton::driver::cu_device device(torch_get_cuda_device(dev_id), false);
    (*fn)((void**)args.c_str(), args.size(), *id_grid_map.at({op_id, dev_id}), &stream, &device);
  }
}


static auto registry = torch::RegisterOperators()
                       .op("triton::launch_kernel", &launch_kernel)
                       .op("triton::cuda_empty_like", &cuda_empty_like)
                       .op("triton::cdiv_sum", &cdiv_sum)
                       .op("triton::synchronize", &synchronize);
