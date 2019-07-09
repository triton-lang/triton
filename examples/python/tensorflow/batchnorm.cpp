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
    const Tensor& x = context->input(0);
    const Tensor& g = context->input(1);
    const Tensor& b = context->input(2);
    // get sizes
    int C = x.dim_size(0);
    int H = x.dim_size(1);
    int W = x.dim_size(2);
    int B = x.dim_size(3);
    // allocate outputs
    Tensor* y = nullptr;
    Tensor* m = nullptr;
    Tensor* v = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, x.shape(), &y));
    OP_REQUIRES_OK(context, context->allocate_output(1, g.shape(), &m));
    OP_REQUIRES_OK(context, context->allocate_output(2, g.shape(), &v));
    // triton handles
    triton::driver::cu_buffer tx(ctx, (CUdeviceptr)x.flat<float>().data(), false);
    triton::driver::cu_buffer tg(ctx, (CUdeviceptr)g.flat<float>().data(), false);
    triton::driver::cu_buffer tb(ctx, (CUdeviceptr)b.flat<float>().data(), false);
    triton::driver::cu_buffer ty(ctx, (CUdeviceptr)y->flat<float>().data(), false);
    triton::driver::cu_buffer tm(ctx, (CUdeviceptr)m->flat<float>().data(), false);
    triton::driver::cu_buffer tv(ctx, (CUdeviceptr)v->flat<float>().data(), false);
    // create config
    triton::dnn::batchnorm_forward batchnorm(C, 1, H, W, B, "fp32");
    std::ostringstream oss;
    batchnorm.src(oss);
    std::string src = oss.str();
    triton::jit jit(ctx);
    jit.add_module("batchnorm", src.c_str(), jit.get_valid("batchnorm", src.c_str()));
    triton::driver::kernel* kernel = jit.get_function("batchnorm");
    size_t TM = jit.get_int("TM");
    triton::jit::launch_information info = jit.get_launch_info("batchnorm");
    batchnorm.enqueue(stream, kernel, &ty, &tm, &tv, &tx, &tg, &tb, TM, info.num_threads);
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
    const Tensor& dy = context->input(0);
    const Tensor&  x = context->input(1);
    const Tensor&  g = context->input(2);
    const Tensor&  m = context->input(3);
    const Tensor&  v = context->input(4);
    // get sizes
    int C = x.dim_size(0);
    int H = x.dim_size(1);
    int W = x.dim_size(2);
    int B = x.dim_size(3);
    // allocate outputs
    Tensor* dx = nullptr;
    Tensor* dg = nullptr;
    Tensor* db = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, x.shape(), &dx));
    OP_REQUIRES_OK(context, context->allocate_output(1, g.shape(), &dg));
    OP_REQUIRES_OK(context, context->allocate_output(2, g.shape(), &db));
    // triton handles
    triton::driver::cu_buffer tdy(ctx, (CUdeviceptr)dy.flat<float>().data(), false);
    triton::driver::cu_buffer tx(ctx, (CUdeviceptr)x.flat<float>().data(), false);
    triton::driver::cu_buffer tg(ctx, (CUdeviceptr)g.flat<float>().data(), false);
    triton::driver::cu_buffer tm(ctx, (CUdeviceptr)m.flat<float>().data(), false);
    triton::driver::cu_buffer tv(ctx, (CUdeviceptr)v.flat<float>().data(), false);
    triton::driver::cu_buffer tdx(ctx, (CUdeviceptr)dx->flat<float>().data(), false);
    triton::driver::cu_buffer tdg(ctx, (CUdeviceptr)dg->flat<float>().data(), false);
    triton::driver::cu_buffer tdb(ctx, (CUdeviceptr)db->flat<float>().data(), false);
    // create config
    triton::dnn::batchnorm_backward batchnorm(C, 1, H, W, B, "fp32");
    std::ostringstream oss;
    batchnorm.src(oss);
    std::string src = oss.str();
    triton::jit jit(ctx);
    jit.add_module("batchnorm", src.c_str(), jit.get_valid("batchnorm", src.c_str()));
    triton::driver::kernel* kernel = jit.get_function("batchnorm");
    size_t TM = jit.get_int("TM");
    triton::jit::launch_information info = jit.get_launch_info("batchnorm");
    batchnorm.enqueue(stream, kernel, &tdx, &tdg, &tdb, &tdy, &tx, &tg, &tm, &tv, TM, info.num_threads);
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
