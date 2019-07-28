#include <iostream>

#include "triton/driver/buffer.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/runtime/jit.h"
#include "triton/tools/bench.hpp"
#include "triton/dnn/conv.h"

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

class Conv2dOp : public OpKernel {
public:
  explicit Conv2dOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context){
    // get device/stream
    GPUDevice device =  context->eigen_device<GPUDevice>();
    triton::driver::cu_stream sstream(device.stream(), false);
    triton::driver::context* ctx = sstream.context();
    triton::driver::stream* stream = &sstream;
    // get inputs
    const Tensor& tfa = context->input(0);
    const Tensor& tfb = context->input(1);
    // get shapes
    int32_t B  = tfa.dim_size(0);
    int32_t Ca = tfa.dim_size(1);
    int32_t D = 1;
    int32_t H  = tfa.dim_size(2);
    int32_t W  = tfa.dim_size(3);
    int32_t Cb = tfb.dim_size(0);
    int32_t T = 1;
    int32_t R  = tfb.dim_size(1);
    int32_t S  = tfb.dim_size(2);
    int32_t NF  = tfb.dim_size(3);
    assert(Ca == Cb);
    int32_t C = Ca;
    int32_t stride_d = 1, stride_h = 1, stride_w = 1;
    int32_t pad_d = 0, pad_h = 0, pad_w = 0;
    bool has_bias = false;
    // wrap buffers
    triton::driver::cu_buffer a(ctx, (CUdeviceptr)tfa.flat<Eigen::half>().data(), false);
    triton::driver::cu_buffer b(ctx, (CUdeviceptr)tfb.flat<Eigen::half>().data(), false);
    triton::driver::buffer* bias = nullptr;
    // template
    triton::dnn::conv conv(B, C,
                           D, H, W,
                           T, R, S,
                           NF,
                           stride_d, stride_h, stride_w,
                           pad_d, pad_h, pad_w,
                           1, 1, 1,
                           "fp16", "fp16",
                           triton::dnn::conv::FPROP, has_bias);
    // allocate output
    auto c_shapes = conv.c_shapes();
    Tensor* tfc = nullptr;
    TensorShape out_shape({c_shapes[0], c_shapes[1], c_shapes[2], c_shapes[3]});
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &tfc));
    triton::driver::cu_buffer c(ctx, (CUdeviceptr)tfc->flat<float>().data(), false);
    // enqueue
    conv.enqueue(stream, {&a, &b, &c, bias});
  }
};

REGISTER_KERNEL_BUILDER(Name("Conv2d").Device(DEVICE_GPU), Conv2dOp);
REGISTER_OP("Conv2d")
    .Input("a: float16")
    .Input("b: float16")
    .Output("c: float32")
;
