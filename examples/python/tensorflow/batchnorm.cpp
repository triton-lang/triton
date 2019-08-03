#include <iostream>

#include "triton/driver/buffer.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/runtime/jit.h"
#include "triton/tools/bench.hpp"
#include "triton/dnn/batchnorm.h"

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using GPUDevice = Eigen::GpuDevice;

class BatchnormForwardOp : public OpKernel {
public:
  explicit BatchnormForwardOp(OpKernelConstruction* context): OpKernel(context) {
    context->GetAttr("eps", &eps_);
  }

  void Compute(OpKernelContext* context){
    // get device/stream
    GPUDevice device =  context->eigen_device<GPUDevice>();
    triton::driver::cu_stream sstream(device.stream(), false);
    triton::driver::context* ctx = sstream.context();
    triton::driver::stream* stream = &sstream;
    // get inputs
    const Tensor& fw_x = context->input(0);
    const Tensor& fw_g = context->input(1);
    const Tensor& fw_b = context->input(2);
    // get sizes
    int C = fw_x.dim_size(0);
    int H = fw_x.dim_size(1);
    int W = fw_x.dim_size(2);
    int B = fw_x.dim_size(3);
    // allocate outputs
    Tensor* fw_y = nullptr;
    Tensor* fw_m = nullptr;
    Tensor* fw_v = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, fw_x.shape(), &fw_y));
    OP_REQUIRES_OK(context, context->allocate_output(1, fw_g.shape(), &fw_m));
    OP_REQUIRES_OK(context, context->allocate_output(2, fw_g.shape(), &fw_v));
    // triton handles
    triton::driver::cu_buffer x(ctx, fw_x.tensor_data().size(), (CUdeviceptr)fw_x.tensor_data().data(), false);
    triton::driver::cu_buffer g(ctx, fw_g.tensor_data().size(), (CUdeviceptr)fw_g.tensor_data().data(), false);
    triton::driver::cu_buffer b(ctx, fw_b.tensor_data().size(), (CUdeviceptr)fw_b.tensor_data().data(), false);
    triton::driver::cu_buffer y(ctx, fw_y->tensor_data().size(), (CUdeviceptr)fw_y->tensor_data().data(), false);
    triton::driver::cu_buffer m(ctx, fw_m->tensor_data().size(), (CUdeviceptr)fw_m->tensor_data().data(), false);
    triton::driver::cu_buffer v(ctx, fw_v->tensor_data().size(), (CUdeviceptr)fw_v->tensor_data().data(), false);
    // create config
    triton::dnn::batchnorm_forward batchnorm(C, 1, H, W, B, "float", triton::dnn::FULL_TUNING);
    batchnorm.enqueue(stream, {&y, &m, &v, &x, &g, &b});
  }

private:
  float eps_;
};


REGISTER_KERNEL_BUILDER(Name("BatchnormForward").Device(DEVICE_GPU), BatchnormForwardOp);
REGISTER_OP("BatchnormForward")
    .Input("x: T")
    .Input("g: float")
    .Input("b: float")
    .Output("y: T")
    .Output("m: float")
    .Output("v: float")
    .Attr("T: {float}")
    .Attr("eps: float")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(0));
      ctx->set_output(1, ctx->input(1));
      ctx->set_output(2, ctx->input(1));
      return Status::OK();
    })
;


class BatchnormBackwardOp : public OpKernel {
public:
  explicit BatchnormBackwardOp(OpKernelConstruction* context): OpKernel(context) {
    context->GetAttr("eps", &eps_);
  }

  void Compute(OpKernelContext* context){
    // get device/stream
    GPUDevice device =  context->eigen_device<GPUDevice>();
    triton::driver::cu_stream sstream(device.stream(), false);
    triton::driver::context* ctx = sstream.context();
    triton::driver::stream* stream = &sstream;
    // get inputs
    const Tensor& fw_dy = context->input(0);
    const Tensor&  fw_x = context->input(1);
    const Tensor&  fw_g = context->input(2);
    const Tensor&  fw_m = context->input(3);
    const Tensor&  fw_v = context->input(4);
    // get sizes
    int C = fw_x.dim_size(0);
    int H = fw_x.dim_size(1);
    int W = fw_x.dim_size(2);
    int B = fw_x.dim_size(3);
    // allocate outputs
    Tensor* fw_dx = nullptr;
    Tensor* fw_dg = nullptr;
    Tensor* fw_db = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, fw_x.shape(), &fw_dx));
    OP_REQUIRES_OK(context, context->allocate_output(1, fw_g.shape(), &fw_dg));
    OP_REQUIRES_OK(context, context->allocate_output(2, fw_g.shape(), &fw_db));
    // triton handles
    triton::driver::cu_buffer dy(ctx, fw_dy.tensor_data().size(), (CUdeviceptr)fw_dy.tensor_data().data(), false);
    triton::driver::cu_buffer x(ctx, fw_x.tensor_data().size(), (CUdeviceptr)fw_x.tensor_data().data(), false);
    triton::driver::cu_buffer g(ctx, fw_g.tensor_data().size(), (CUdeviceptr)fw_g.tensor_data().data(), false);
    triton::driver::cu_buffer m(ctx, fw_m.tensor_data().size(), (CUdeviceptr)fw_m.tensor_data().data(), false);
    triton::driver::cu_buffer v(ctx, fw_v.tensor_data().size(), (CUdeviceptr)fw_v.tensor_data().data(), false);
    triton::driver::cu_buffer dx(ctx, fw_dx->tensor_data().size(), (CUdeviceptr)fw_dx->tensor_data().data(), false);
    triton::driver::cu_buffer dg(ctx, fw_dg->tensor_data().size(), (CUdeviceptr)fw_dg->tensor_data().data(), false);
    triton::driver::cu_buffer db(ctx, fw_db->tensor_data().size(), (CUdeviceptr)fw_db->tensor_data().data(), false);
    // create config
    triton::dnn::batchnorm_backward batchnorm(C, 1, H, W, B, "float", triton::dnn::FULL_TUNING);
    batchnorm.enqueue(stream, {&dx, &dg, &db, &dy, &x, &g, &m, &v});
  }

private:
  float eps_;
};


REGISTER_KERNEL_BUILDER(Name("BatchnormBackward").Device(DEVICE_GPU), BatchnormBackwardOp);
REGISTER_OP("BatchnormBackward")
    .Input("dy: TY")
    .Input("x: TX")
    .Input("g: float")
    .Input("m: float")
    .Input("v: float")
    .Output("dx: TY")
    .Output("dg: float")
    .Output("db: float")
    .Attr("TX: {float}")
    .Attr("TY: {float}")
    .Attr("eps: float")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(1));
      ctx->set_output(1, ctx->input(2));
      ctx->set_output(2, ctx->input(2));
      return Status::OK();
    })
;
