#include <iostream>

#include "triton/driver/buffer.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/runtime/jit.h"
#include "triton/tools/bench.hpp"
#include "triton/dnn/gemm.h"

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

class DotOp : public OpKernel {
 public:
  explicit DotOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context){
    // get device/stream
    GPUDevice device =  context->eigen_device<GPUDevice>();
    triton::driver::cu_stream sstream(device.stream(), false);
    triton::driver::context* ctx = sstream.context();
    triton::driver::stream* stream = &sstream;
    // get inputs
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    const Tensor& locks = context->input(2);
    // get shapes
    const int32_t M = a.dim_size(0);
    const int32_t N = b.dim_size(0);
    const int32_t K = a.dim_size(1);
    // allocate output
    Tensor* c = nullptr;
    TensorShape out_shape({(int64)M, (int64)N});
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &c));
    // return early if possible
    if (out_shape.num_elements() == 0)
      return;
    // initialize default compute device
    triton::jit jit(ctx);
    // matrix multiplication parameters
    triton::driver::cu_buffer da(ctx, (CUdeviceptr)a.flat<Eigen::half>().data(), false);
    triton::driver::cu_buffer db(ctx, (CUdeviceptr)b.flat<Eigen::half>().data(), false);
    triton::driver::cu_buffer dc(ctx, (CUdeviceptr)c->flat<float>().data(), false);
    triton::driver::cu_buffer dlocks(ctx, (CUdeviceptr)locks.flat<int32_t>().data(), false);
    // benchmark a given matrix multiplication kernel
    auto benchmark = [&](triton::driver::kernel* kernel,
                         triton::jit::launch_information info) {
      // launch info
      unsigned TM = info.global_range_size[0];
      unsigned TN = info.global_range_size[1];
      unsigned nthreads = info.num_threads;
      unsigned GZ = jit.get_int("GZ");
      std::array<size_t, 3> grid = {(M + TM - 1)/TM, (N + TN - 1)/TN, GZ};
      triton::dnn::gemm::set_arg(kernel, &da, &db, &dc, M, N, K, &dlocks, grid[0], grid[1]);
      stream->enqueue(kernel, grid, {nthreads, 1, 1});
      stream->synchronize();
      double ts = triton::tools::bench([&](){stream->enqueue(kernel, grid, {nthreads, 1, 1});},
                        [&](){ stream->synchronize(); }, ctx->device());
      return  2.*M*N*K / ts * 1e-3;
    };
    std::string src = triton::dnn::gemm::src(false, true, "fp16", "fp16", 1, 1);
//     just-in-time compile source-code
    jit.autotune("matmul", src.c_str(), benchmark);
//    jit.add_module("matmul", src.c_str(), {4, 2, 8, 4, 2, 32, 1, 4, 1, 1, 8, 8, 8, 1});
//    jit.add_module("matmul", src.c_str(), {16, 4, 128, 16, 4, 128, 2, 2, 2, 2, 8, 32, 8, 1});
//    jit.add_module("matmul", src.c_str(), {8, 8, 128, 16, 8, 128, 2, 2, 2, 2, 16, 32, 8, 1 });
//    jit.add_module("matmul", src.c_str(), {16, 4, 128, 16, 4, 128, 2, 2, 2, 2, 8, 16, 8, 1});
    jit.add_module("matmul", src.c_str(), {16, 2, 128, 32, 32, 2, 2, 2, 2, 8, 8, 4, 2, 1}); //NN
    triton::driver::kernel* kernel = jit.get_function("matmul");
    triton::jit::launch_information info = jit.get_launch_info("matmul");
    std::cout << benchmark(kernel, info) << std::endl;;
  }

private:
};

REGISTER_KERNEL_BUILDER(Name("Dot").Device(DEVICE_GPU), DotOp);
REGISTER_OP("Dot")
    .Input("a: float16")
    .Input("b: float16")
    .Input("locks: int32")
    .Output("c: float32")
;
