#include "triton/driver/buffer.h"
#include "triton/driver/stream.h"
#include "triton/runtime/function.h"
#include "triton/tools/bench.hpp"
#include "torch/script.h"
#include "ATen/cuda/CUDAContext.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x);

namespace rt = triton::runtime;
namespace drv = triton::driver;

typedef std::pair<size_t, size_t> map_key_t;
extern std::map<map_key_t, std::shared_ptr<rt::function::grid_fn_ty>> id_grid_map;
extern std::map<map_key_t, std::shared_ptr<rt::function>> id_fn_map;

void launch_kernel(int64_t op_id, int64_t dev_id, const std::string& args){
  CUstream custream = (CUstream)at::cuda::getCurrentCUDAStream(dev_id).stream();
  triton::driver::cu_stream stream(custream, false);
  triton::driver::context* ctx = stream.context();
  (*id_fn_map.at({op_id, dev_id}))((void**)args.c_str(), args.size(), *id_grid_map.at({op_id, dev_id}), &stream);
}


static auto registry = torch::RegisterOperators("triton::launch_kernel", &launch_kernel);
