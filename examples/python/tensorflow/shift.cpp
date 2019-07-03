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

class ShiftConvOp : public OpKernel {
public:
  explicit ShiftConvOp(OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr("shift_h", &h_shift_h_);
    context->GetAttr("shift_w", &h_shift_w_);
    R_ = 3;
    S_ = 3;
  }

  void ComputeCommon(OpKernelContext* context){

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
    // shapes for a
    int64_t Ca  = tf_a.dim_size(0);
    int64_t H   = tf_a.dim_size(1);
    int64_t W   = tf_a.dim_size(2);
    int64_t B   = tf_a.dim_size(3);
    // shapes for b
    int64_t Cb  = tf_b.dim_size(0);
    int64_t F   = tf_b.dim_size(1);
    // checks
    OP_REQUIRES(context, Ca == Cb, tensorflow::errors::InvalidArgument("operands must have the same number of channels"));
    int64_t C = Ca;
    // shapes for c
    Tensor* tf_c = nullptr;
    TensorShape out_shape({Ca, H, W, B});
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &tf_c));
    // return early if possible
    if (out_shape.num_elements() == 0)
      return;
    // initialize default compute device
    triton::jit jit(ctx);
    // matrix multiplication parameters
    triton::driver::cu_buffer da(ctx,      (CUdeviceptr)tf_a.flat<float>().data(), false);
    triton::driver::cu_buffer db(ctx,      (CUdeviceptr)tf_b.flat<float>().data(), false);
    triton::driver::cu_buffer dc(ctx,      (CUdeviceptr)tf_c->flat<float>().data(), false);
    // shift configuration
    int32_t* shift_h_data = h_shift_h_.flat<int32_t>().data();
    int32_t* shift_w_data = h_shift_w_.flat<int32_t>().data();
    std::vector<int32_t> shift_h(shift_h_data, shift_h_data + C);
    std::vector<int32_t> shift_w(shift_w_data, shift_w_data + C);
    triton::dnn::shift shift(B, C, 1, H, W, 1, R_, S_, F, shift_h, shift_w, "fp32", "fp32", triton::dnn::shift::FPROP, false);
    // benchmark a given matrix multiplication kernel
    auto benchmark = [&](triton::driver::kernel* kernel,
                         triton::jit::launch_information info) {
      // launch info
      unsigned TM = info.global_range_size[0];
      unsigned TN = info.global_range_size[1];
      unsigned nthreads = info.num_threads;
      shift.init(stream, (triton::driver::cu_module*)kernel->module());
      shift.enqueue(stream, kernel, &da, &db, &dc, TM, TN, nthreads);
      stream->synchronize();
      double ts = triton::tools::bench([&](){ shift.enqueue(stream, kernel, &da, &db, &dc, TM, TN, nthreads); },
                        [&](){ stream->synchronize(); }, ctx->device());
      return  shift.get_nflops() / ts * 1e-3;
    };

    std::ostringstream oss;
    shift.src(oss);
    std::string src = oss.str();
    triton::jit::tune_res_t best = jit.autotune("shift", src.c_str(), benchmark);
  }

private:
  Tensor  h_shift_h_;
  Tensor  h_shift_w_;
//  triton::driver::buffer* d_shift_h_;
//  triton::driver::buffer* d_shift_w_;
  int R_;
  int S_;
};

REGISTER_KERNEL_BUILDER(Name("ShiftConv").Device(DEVICE_GPU), ShiftConvOp);
REGISTER_OP("ShiftConv")
    .Input("a: float32")
    .Input("b: float32")
    .Attr("shift_h: tensor")
    .Attr("shift_w: tensor")
    .Output("c: float32")
;
