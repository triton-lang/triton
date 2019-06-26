#include <iostream>

#include "triton/driver/buffer.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/runtime/jit.h"
#include "triton/tools/bench.hpp"
#include "triton/dnn/gemm.h"
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

//torch::Tensor conv_common(
//    int32_t B, int32_t C, int32_t D, int32_t H, int32_t W,
//    int32_t T, int32_t R, int32_t S, int32_t NF,
//    int32_t stride_d, int32_t stride_h, int32_t stride_w,
//    int32_t pad_d, int32_t pad_h, int32_t pad_w,
//    triton::dnn::conv::type ty,
//    torch::Tensor torcha, torch::Tensor torchb, torch::Tensor torchbias,
//    bool autotune = false
//    ) {

//}

class DenseConvOp : public OpKernel {
 public:
  explicit DenseConvOp(OpKernelConstruction* context) : OpKernel(context) {
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

    // get conv configuration
    triton::dnn::conv configuration(B, C, D, H, W, T, R, S, NF,
                                    stride_d, stride_h, stride_w,
                                    pad_d, pad_h, pad_w,
                                    1, 1, 1,
                                    triton::dnn::conv::FPROP, has_bias);

    // Bind memory
    triton::driver::cu_buffer a(ctx, (CUdeviceptr)tfa.flat<float>().data(), false);
    triton::driver::cu_buffer b(ctx, (CUdeviceptr)tfb.flat<float>().data(), false);
//    triton::driver::cu_buffer cubias(ctx, (CUdeviceptr)torchbias.storage().data(), false);
//    triton::driver::buffer* bias = has_bias ? &cubias : nullptr;
    triton::driver::buffer* bias = nullptr;

    // allocate output
    auto c_shapes = configuration.c_shapes();
    Tensor* tfc = nullptr;
    TensorShape out_shape({c_shapes[0], c_shapes[1], c_shapes[2], c_shapes[3]});
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &tfc));
    triton::driver::cu_buffer c(ctx, (CUdeviceptr)tfc->flat<float>().data(), false);

    // benchmark a given convolution kernel
    triton::jit jit(ctx);
    auto benchmark = [&](triton::driver::kernel* kernel,
                         triton::jit::launch_information info) {
      configuration.init(stream, (triton::driver::cu_module*)kernel->module());
      unsigned TM = info.global_range_size[0];
      unsigned TN = info.global_range_size[1];
      unsigned nthreads = info.num_threads;
      unsigned GZ = jit.get_int("GZ");
      configuration.enqueue(stream, kernel, &a, &b, &c, bias, TM, TN, GZ, nthreads);
      stream->synchronize();
      double ts = triton::tools::bench([&](){ configuration.enqueue(stream, kernel, &a, &b, &c, bias, TM, TN, GZ, nthreads); },
                        [&](){ stream->synchronize(); }, stream->context()->device());
      return configuration.get_nflops() / ts * 1e-3;
    };

    std::ostringstream oss;
    configuration.src(oss);
    std::string src = oss.str();

    triton::jit::tune_res_t best = jit.autotune("conv", src.c_str(), benchmark);
    jit.add_module("conv", src.c_str(), best.params);
  }
};

REGISTER_KERNEL_BUILDER(Name("DenseConv").Device(DEVICE_GPU), DenseConvOp);
REGISTER_OP("DenseConv")
    .Input("a: float32")
    .Input("b: float32")
    .Output("c: float32")
;
