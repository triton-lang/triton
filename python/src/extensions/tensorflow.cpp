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

isaac::DType sc_dtype(DataType dtype){
  switch(dtype){
    case DT_FLOAT: return isaac::FLOAT_TYPE;
    default: throw std::runtime_error("DataType not supported");
  }
}

REGISTER_OP("Conv2d")
    .Attr("T: {float}")
    .Input("input: T")
    .Input("filter: T")
    .Attr("strides: list(int)")
    .Attr("data_format: string")
    .Attr("padding:  string")
    .Output("output: T");


REGISTER_OP("Conv3d")
    .Attr("T: {float}")
    .Input("input: T")
    .Input("filter: T")
    .Attr("strides: list(int)")
    .Attr("data_format: string")
    .Attr("padding:  string")
    .Output("output: T");

template<size_t DIM>
class ConvOp : public OpKernel {
 public:
  explicit ConvOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get attributes
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_), errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &pad_));
  }

  void Compute(OpKernelContext* context){
    GPUDevice device =  context->eigen_device<GPUDevice>();
    drv::Stream stream(device.stream(), false);

    /* Get inputs */
    const Tensor& inputs = context->input(0);
    const Tensor& filter = context->input(1);

    /* Get dtype */
    isaac::DType dtype = sc_dtype(inputs.dtype());

    /* Extract shapes */
    // Input
    int64 D = 1, H = 1, W = 1;
    const int64 N = inputs.dim_size(0);
    const int64 Ci = inputs.dim_size(1);
    if(DIM > 2) D = inputs.dim_size(2);
    if(DIM > 1) H = inputs.dim_size(2 + (DIM > 2));
    if(DIM > 0) W = inputs.dim_size(2 + (DIM > 2) + (DIM > 1));
    // Filter
    int64 T = 1, R = 1, S = 1;
    const int64 Cf = filter.dim_size(0);
    if(DIM > 2) T = filter.dim_size(1);
    if(DIM > 1) R = filter.dim_size(1 + (DIM > 2));
    if(DIM > 0) S = filter.dim_size(1 + (DIM > 2) + (DIM > 1));
    const int64 K = filter.dim_size(1 + DIM);
    // Strides
    int64_t stride_d = 1, stride_h = 1, stride_w = 1;
    if(DIM > 2) stride_d = strides_[2];
    if(DIM > 1) stride_h = strides_[2 + (DIM > 2)];
    if(DIM > 0) stride_w = strides_[2 + (DIM > 2) + (DIM > 1)];
    // Output
    int64 M, P, Q, pad_d, pad_h, pad_w;
    GetWindowedOutputSize(D, T, stride_d, pad_, &M, &pad_d);
    GetWindowedOutputSize(H, R, stride_h, pad_, &P, &pad_h);
    GetWindowedOutputSize(W, S, stride_w, pad_, &Q, &pad_w);


    /* Requirements */
    OP_REQUIRES(context, Ci==Cf, errors::InvalidArgument("input and filter must have the same depth: ", Ci, " vs ", Cf));
    const int64_t C = Ci;

    /* Allocate output */
    Tensor* output = nullptr;
    TensorShape out_shape;
    if(DIM==3) out_shape = TensorShape({N, K, M, P, Q});
    else if(DIM == 2) out_shape = TensorShape({N, K, P, Q});
    else if(DIM == 1) out_shape = TensorShape({N, K, Q});
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    /* Return early if possible */
    if (out_shape.num_elements() == 0)
      return;

    /* Compute convolution */
    isaac::driver::Buffer I(stream.context(), (CUdeviceptr)inputs.flat<float>().data(), false);
    isaac::driver::Buffer F(stream.context(), (CUdeviceptr)filter.flat<float>().data(), false);
    isaac::driver::Buffer O(stream.context(), (CUdeviceptr)output->flat<float>().data(), false);
    isaac::CONV(stream.context().device(), stream, dtype,
                N, K, M, P, Q, C, T, R, S, D, H, W, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, I, F, O);
  }

private:
  Padding pad_;
  TensorFormat data_format_;
  std::vector<int32> strides_;
};

REGISTER_KERNEL_BUILDER(Name("Conv2d").Device(DEVICE_GPU), ConvOp<2>);
REGISTER_KERNEL_BUILDER(Name("Conv3d").Device(DEVICE_GPU), ConvOp<3>);
