#include <torch/torch.h>
#include <torch/script.h>
#include "ATen/cuda/CUDAContext.h"
#include "triton/driver/stream.h"
#include "triton/dnn/batchnorm.h"

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor>
      batchnorm_ymv(const torch::Tensor fw_x,
                    const torch::Tensor fw_g,
                    const torch::Tensor fw_b,
                    double eps) {
  CHECK_INPUT(fw_x);
  CHECK_INPUT(fw_g);
  CHECK_INPUT(fw_b);
  // Wrap CUDA handles
  c10::DeviceIndex device = fw_x.storage().device().index();
  CUstream custream = (CUstream)at::cuda::getCurrentCUDAStream(device).stream();
  triton::driver::cu_stream stream(custream, false);
  triton::driver::context* ctx = stream.context();
  // get sizes
  int C = fw_x.size(0);
  int H = fw_x.size(1);
  int W = fw_x.size(2);
  int B = fw_x.size(3);
  // allocate outputs
  torch::Tensor fw_y = torch::empty(fw_x.sizes()).cuda();
  torch::Tensor fw_m = torch::empty(fw_g.sizes()).cuda();
  torch::Tensor fw_v = torch::empty(fw_g.sizes()).cuda();
  triton::driver::cu_buffer x(ctx, (CUdeviceptr)fw_x.storage().data(), false);
  triton::driver::cu_buffer g(ctx, (CUdeviceptr)fw_g.storage().data(), false);
  triton::driver::cu_buffer b(ctx, (CUdeviceptr)fw_b.storage().data(), false);
  triton::driver::cu_buffer y(ctx, (CUdeviceptr)fw_y.storage().data(), false);
  triton::driver::cu_buffer m(ctx, (CUdeviceptr)fw_m.storage().data(), false);
  triton::driver::cu_buffer v(ctx, (CUdeviceptr)fw_v.storage().data(), false);
  // create template
  triton::dnn::batchnorm_forward batchnorm(C, 1, H, W, B, "float");
  batchnorm.enqueue(&stream, {&y, &m, &v, &x, &g, &b});
  stream.synchronize();
  return {fw_y, fw_m, fw_v};
}

std::vector<torch::Tensor>
      batchnorm_dxdgdb(const torch::Tensor fw_dy,
                       const torch::Tensor fw_x,
                       const torch::Tensor fw_g,
                       const torch::Tensor fw_m,
                       const torch::Tensor fw_v,
                       double eps) {
  CHECK_INPUT(fw_dy);
  CHECK_INPUT(fw_x);
  CHECK_INPUT(fw_g);
  CHECK_INPUT(fw_m);
  CHECK_INPUT(fw_v);
  // Wrap CUDA handles
  c10::DeviceIndex device = fw_x.storage().device().index();
  CUstream custream = (CUstream)at::cuda::getCurrentCUDAStream(device).stream();
  triton::driver::cu_stream stream(custream, false);
  triton::driver::context* ctx = stream.context();
  // get sizes
  int C = fw_x.size(0);
  int H = fw_x.size(1);
  int W = fw_x.size(2);
  int B = fw_x.size(3);
  // allocate outputs
  torch::Tensor fw_dx = torch::empty(fw_x.sizes()).cuda();
  torch::Tensor fw_dg = torch::empty(fw_g.sizes()).cuda();
  torch::Tensor fw_db = torch::empty(fw_g.sizes()).cuda();
  // triton handles
  triton::driver::cu_buffer dy(ctx, (CUdeviceptr)fw_dy.storage().data(), false);
  triton::driver::cu_buffer x(ctx,  (CUdeviceptr) fw_x.storage().data(), false);
  triton::driver::cu_buffer g(ctx,  (CUdeviceptr) fw_g.storage().data(), false);
  triton::driver::cu_buffer m(ctx,  (CUdeviceptr) fw_m.storage().data(), false);
  triton::driver::cu_buffer v(ctx,  (CUdeviceptr) fw_v.storage().data(), false);
  triton::driver::cu_buffer dx(ctx, (CUdeviceptr)fw_dx.storage().data(), false);
  triton::driver::cu_buffer dg(ctx, (CUdeviceptr)fw_dg.storage().data(), false);
  triton::driver::cu_buffer db(ctx, (CUdeviceptr)fw_db.storage().data(), false);
  // create config
  triton::dnn::batchnorm_backward batchnorm(C, 1, H, W, B, "float", eps);
  batchnorm.enqueue(&stream, {&dx, &dg, &db, &dy, &x, &g, &m, &v});
  stream.synchronize();
  return {fw_dx, fw_dg, fw_db};
}

static auto registry =
  torch::jit::RegisterOperators("triton::batchnorm_ymv", &batchnorm_ymv)
                            .op("triton::batchnorm_dxdgdb", &batchnorm_dxdgdb);
