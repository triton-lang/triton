#include <iostream>

#include "isaac/driver/buffer.h"
#include "isaac/driver/backend.h"
#include "isaac/driver/stream.h"
#include "isaac/api.h"

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
namespace drv = isaac::driver;

REGISTER_OP("Conv")
    .Attr("T: {float}")
    .Input("input: T")
    .Input("filter: T")
    .Attr("strides: list(int)")
    .Attr("data_format: string")
    .Attr("padding:  string")
    .Output("output: T");


class ConvOp : public OpKernel {
 public:
  explicit ConvOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context){
    GPUDevice device =  context->eigen_device<GPUDevice>();
    drv::Stream stream(device.stream(), false);
    isaac::DType dtype = isaac::FLOAT_TYPE;

    const Tensor& inputs = context->input(0);
    const Tensor& filter = context->input(1);


    const int64 N = inputs.dim_size(0);
    const int64 D = 1;
    const int64 Ci = inputs.dim_size(1);
    const int64 H = inputs.dim_size(2);
    const int64 W = inputs.dim_size(3);

    const int64 T = 1;
    const int64 Cf = filter.dim_size(0);
    const int64 R = filter.dim_size(1);
    const int64 S = filter.dim_size(2);
    const int64 K = filter.dim_size(3);

    assert(Ci == Cf);
    const int64_t C = Ci;

    const int64 stride_d = 1;
    const int64 stride_h = 1;
    const int64 stride_w = 1;
    int64 M = 1, P, Q, pad_d, pad_h, pad_w;

    Padding pad = SAME;
    TensorFormat data_format = FORMAT_NCHW;
    GetWindowedOutputSize(M, T, stride_d, pad, &M, &pad_d);
    GetWindowedOutputSize(H, R, stride_h, pad, &P, &pad_h);
    GetWindowedOutputSize(W, S, stride_w, pad, &Q, &pad_w);

    Tensor* output = nullptr;
    TensorShape out_shape = ShapeFromFormat(data_format, N, P, Q, K);
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    if (out_shape.num_elements() == 0)
      return;

    isaac::scalar alpha(1., dtype), beta(0., dtype);
    isaac::driver::Buffer I(stream.context(), (CUdeviceptr)inputs.flat<float>().data(), false);
    isaac::driver::Buffer F(stream.context(), (CUdeviceptr)filter.flat<float>().data(), false);
    isaac::driver::Buffer O(stream.context(), (CUdeviceptr)output->flat<float>().data(), false);

    isaac::CONV(stream.context().device(), stream, dtype,
                N, K, M, P, Q, C, T, R, S, D, H, W, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, alpha, I, F, beta, O);
  }

};

REGISTER_KERNEL_BUILDER(Name("Conv").Device(DEVICE_GPU), ConvOp);
