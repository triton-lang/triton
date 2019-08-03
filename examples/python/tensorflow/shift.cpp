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

template<triton::dnn::op_t OP>
class ShiftConvOp : public OpKernel {
public:
  explicit ShiftConvOp(OpKernelConstruction* context) : OpKernel(context), layout_(triton::dnn::NCHW) {
    context->GetAttr("shift_h", &h_shift_h_);
    context->GetAttr("shift_w", &h_shift_w_);
    context->GetAttr("stride_h", &stride_h_);
    context->GetAttr("stride_w", &stride_w_);
    R_ = 3;
    S_ = 3;
  }

  void ExtractShapes(const Tensor &x, int64_t &C, int64_t &H, int64_t &W, int64_t &B) {
    if(layout_ == triton::dnn::CHWN){
      C  = x.dim_size(0);
      H  = x.dim_size(1);
      W  = x.dim_size(2);
      B  = x.dim_size(3);
    }
    else if(layout_ == triton::dnn::NCHW){
      B  = x.dim_size(0);
      C  = x.dim_size(1);
      H  = x.dim_size(2);
      W  = x.dim_size(3);
    }
    else{
      throw std::runtime_error("unsupported layout");
    }
  }

  void FillShapes(OpKernelContext* context,
                  int64_t &C, int64_t &H, int64_t &W, int64_t &B, int64_t &F,
                  const Tensor& tf_a, const Tensor& tf_b) {
    if(OP == triton::dnn::WGRAD) {
      int64_t Ha, Wa, Ba;
      int64_t Hb, Wb, Bb;
      ExtractShapes(tf_a, F, Ha, Wa, Ba);
      ExtractShapes(tf_b, C, Hb, Wb, Bb);
      OP_REQUIRES(context, Ha*stride_h_ == Hb, tensorflow::errors::InvalidArgument("operands must have the same image height"));
      OP_REQUIRES(context, Wa*stride_w_ == Wb, tensorflow::errors::InvalidArgument("operands must have the same image width"));
      OP_REQUIRES(context, Ba == Bb, tensorflow::errors::InvalidArgument("operands must have the same batch size"));
      H = Hb;
      W = Wb;
      B = Bb;
    }
    else {
      // shapes for a
      int64_t Ca;
      ExtractShapes(tf_a, Ca, H, W, B);
      if(OP == triton::dnn::BPROP){
        H *= stride_h_;
        W *= stride_w_;
      }
      // shapes for b
      int64_t Cb  = tf_b.dim_size(0);
      F   = tf_b.dim_size(1);
      if(OP == triton::dnn::BPROP)
        std::swap(Cb, F);
      // checks
      OP_REQUIRES(context, Ca == Cb, tensorflow::errors::InvalidArgument("operands must have the same number of channels"));
      C = Ca;
      if(OP == triton::dnn::BPROP)
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
                             "half", "half", OP, has_bias, layout_);

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
    triton::driver::cu_buffer da(ctx, tf_a.tensor_data().size(), (CUdeviceptr)tf_a.tensor_data().data(), false);
    triton::driver::cu_buffer db(ctx, tf_b.tensor_data().size(), (CUdeviceptr)tf_b.tensor_data().data(), false);
    triton::driver::cu_buffer dc(ctx, tf_c->tensor_data().size(), (CUdeviceptr)tf_c->tensor_data().data(), false);
    shift.enqueue(stream, {&da, &db, &dc}, triton::dnn::PARTIAL_TUNING);
  }

private:
  Tensor  h_shift_h_;
  Tensor  h_shift_w_;
  int stride_h_;
  int stride_w_;
  int R_;
  int S_;
  triton::dnn::layout_t layout_;
};

REGISTER_KERNEL_BUILDER(Name("ShiftConv").Device(DEVICE_GPU), ShiftConvOp<triton::dnn::FPROP>);
REGISTER_OP("ShiftConv")
    .Input("a: float16")
    .Input("b: float16")
    .Attr("shift_h: tensor")
    .Attr("shift_w: tensor")
    .Attr("stride_h: int")
    .Attr("stride_w: int")
    .Output("c: float16");

REGISTER_KERNEL_BUILDER(Name("ShiftConvDx").Device(DEVICE_GPU), ShiftConvOp<triton::dnn::BPROP>);
REGISTER_OP("ShiftConvDx")
    .Input("a: float16")
    .Input("b: float16")
    .Attr("shift_h: tensor")
    .Attr("shift_w: tensor")
    .Attr("stride_h: int")
    .Attr("stride_w: int")
    .Output("c: float16");

REGISTER_KERNEL_BUILDER(Name("ShiftConvDw").Device(DEVICE_GPU), ShiftConvOp<triton::dnn::WGRAD>);
REGISTER_OP("ShiftConvDw")
    .Input("a: float16")
    .Input("b: float16")
    .Attr("shift_h: tensor")
    .Attr("shift_w: tensor")
    .Attr("stride_h: int")
    .Attr("stride_w: int")
    .Output("c: float16");

