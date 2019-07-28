#include <iostream>

#include "triton/driver/buffer.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/runtime/jit.h"
#include "triton/tools/bench.hpp"
#include "triton/dnn/dot.h"

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
    // matrix multiplication parameters
    triton::driver::cu_buffer da(ctx, (CUdeviceptr)a.flat<Eigen::half>().data(), false);
    triton::driver::cu_buffer db(ctx, (CUdeviceptr)b.flat<Eigen::half>().data(), false);
    triton::driver::cu_buffer dc(ctx, (CUdeviceptr)c->flat<float>().data(), false);
    // template
    triton::dnn::dot dot(M, N, K, false, false, "fp16", "fp16", 8, 8);
    dot.enqueue(stream, {&da, &db, &dc});
  }

private:
};

REGISTER_KERNEL_BUILDER(Name("Dot").Device(DEVICE_GPU), DotOp);
REGISTER_OP("Dot")
    .Input("a: float16")
    .Input("b: float16")
    .Output("c: float32")
;
