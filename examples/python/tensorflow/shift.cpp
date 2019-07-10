#include <iostream>

#include "triton/driver/buffer.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/runtime/jit.h"
#include "triton/tools/bench.hpp"
#include "triton/dnn/shift.h"

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

template<triton::dnn::shift::type OP>
class ShiftConvOp : public OpKernel {
public:
  explicit ShiftConvOp(OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr("shift_h", &h_shift_h_);
    context->GetAttr("shift_w", &h_shift_w_);
    context->GetAttr("stride_h", &stride_h_);
    context->GetAttr("stride_w", &stride_w_);
    R_ = 3;
    S_ = 3;
  }

  void FillShapes(OpKernelContext* context,
                  int64_t &C, int64_t &H, int64_t &W, int64_t &B, int64_t &F,
                  const Tensor& tf_a, const Tensor& tf_b) {
    if(OP == triton::dnn::shift::WGRAD) {
      // shapes for a
      F  = tf_a.dim_size(0);
      int64_t Ha   = tf_a.dim_size(1);
      int64_t Wa   = tf_a.dim_size(2);
      int64_t Ba   = tf_a.dim_size(3);
      // shapes for b
      C = tf_b.dim_size(0);
      int64_t Hb    = tf_b.dim_size(1);
      int64_t Wb    = tf_b.dim_size(2);
      int64_t Bb    = tf_b.dim_size(3);
      OP_REQUIRES(context, Ha*stride_h_ == Hb, tensorflow::errors::InvalidArgument("operands must have the same image height"));
      OP_REQUIRES(context, Wa*stride_w_ == Wb, tensorflow::errors::InvalidArgument("operands must have the same image width"));
      OP_REQUIRES(context, Ba == Bb, tensorflow::errors::InvalidArgument("operands must have the same batch size"));
      H = Hb;
      W = Wb;
      B = Bb;
    }
    else {
      // shapes for a
      int64_t Ca  = tf_a.dim_size(0);
      H   = tf_a.dim_size(1);
      W   = tf_a.dim_size(2);
      B   = tf_a.dim_size(3);
      if(OP == triton::dnn::shift::BPROP){
        H *= stride_h_;
        W *= stride_w_;
      }
      // shapes for b
      int64_t Cb  = tf_b.dim_size(0);
      F   = tf_b.dim_size(1);
      if(OP == triton::dnn::shift::BPROP)
        std::swap(Cb, F);
      // checks
      OP_REQUIRES(context, Ca == Cb, tensorflow::errors::InvalidArgument("operands must have the same number of channels"));
      C = Ca;
      if(OP == triton::dnn::shift::BPROP)
        std::swap(C, F);
    }

  }

  void Compute(OpKernelContext* context){
    // get device/stream
    GPUDevice device =  context->eigen_device<GPUDevice>();
    triton::driver::cu_stream sstream(device.stream(), false);
    triton::driver::context* ctx = sstream.context();
    triton::driver::stream* stream = &sstream;
    // get inputs
    const Tensor& tf_a = context->input(0);
    const Tensor& tf_b = context->input(1);
    // shapes
    int64_t C, H, W, B, F;
    FillShapes(context, C, H, W, B, F, tf_a, tf_b);
    int64_t D = 1, T = 1;
    bool has_bias = false;
    // shift offsets
    int32_t* shift_h_data = h_shift_h_.flat<int32_t>().data();
    int32_t* shift_w_data = h_shift_w_.flat<int32_t>().data();
    // create configuration
    triton::dnn::shift shift(B, C, D, H, W, T, R_, S_, F,
                             stride_h_, stride_w_,
                             shift_h_data, shift_w_data,
                             "fp32", "fp32", OP, has_bias);

    // shapes for c
    std::vector<int64> c_shapes;
    for(int32_t x: shift.c_shapes())
      c_shapes.push_back(x);
    TensorShape out_shapes(c_shapes);
    Tensor* tf_c = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shapes, &tf_c));
    // return early if possible
    if (out_shapes.num_elements() == 0)
      return;
    // matrix multiplication parameters
    triton::driver::cu_buffer da(ctx,      (CUdeviceptr)tf_a.flat<float>().data(), false);
    triton::driver::cu_buffer db(ctx,      (CUdeviceptr)tf_b.flat<float>().data(), false);
    triton::driver::cu_buffer dc(ctx,      (CUdeviceptr)tf_c->flat<float>().data(), false);
    shift.enqueue(stream, {&da, &db, &dc});
  }

private:
  Tensor  h_shift_h_;
  Tensor  h_shift_w_;
  int stride_h_;
  int stride_w_;
  int R_;
  int S_;
};

REGISTER_KERNEL_BUILDER(Name("ShiftConv").Device(DEVICE_GPU), ShiftConvOp<triton::dnn::shift::FPROP>);
REGISTER_OP("ShiftConv")
    .Input("a: float32")
    .Input("b: float32")
    .Attr("shift_h: tensor")
    .Attr("shift_w: tensor")
    .Attr("stride_h: int")
    .Attr("stride_w: int")
    .Output("c: float32");

REGISTER_KERNEL_BUILDER(Name("ShiftConvDx").Device(DEVICE_GPU), ShiftConvOp<triton::dnn::shift::BPROP>);
REGISTER_OP("ShiftConvDx")
    .Input("a: float32")
    .Input("b: float32")
    .Attr("shift_h: tensor")
    .Attr("shift_w: tensor")
    .Attr("stride_h: int")
    .Attr("stride_w: int")
    .Output("c: float32");

REGISTER_KERNEL_BUILDER(Name("ShiftConvDw").Device(DEVICE_GPU), ShiftConvOp<triton::dnn::shift::WGRAD>);
REGISTER_OP("ShiftConvDw")
    .Input("a: float32")
    .Input("b: float32")
    .Attr("shift_h: tensor")
    .Attr("shift_w: tensor")
    .Attr("stride_h: int")
    .Attr("stride_w: int")
    .Output("c: float32");

