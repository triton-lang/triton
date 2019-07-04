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

typedef std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t,
                   int32_t, int32_t, int32_t, int32_t,
                   int32_t*, int32_t*,
                   triton::dnn::shift::type, bool> shift_key_t;

static std::map<CUstream, std::unique_ptr<triton::driver::stream>> m_stream;
static std::map<shift_key_t, std::unique_ptr<triton::jit>>          m_jit;
static std::map<shift_key_t, std::unique_ptr<triton::dnn::shift>>   m_config;

template<triton::dnn::shift::type OP>
class ShiftConvOp : public OpKernel {
public:
  explicit ShiftConvOp(OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr("shift_h", &h_shift_h_);
    context->GetAttr("shift_w", &h_shift_w_);
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
      OP_REQUIRES(context, Ha == Hb, tensorflow::errors::InvalidArgument("operands must have the same image height"));
      OP_REQUIRES(context, Wa == Wb, tensorflow::errors::InvalidArgument("operands must have the same image width"));
      OP_REQUIRES(context, Ba == Bb, tensorflow::errors::InvalidArgument("operands must have the same batch size"));
      H = Ha;
      W = Wa;
      B = Ba;
    }
    else {
      // shapes for a
      int64_t Ca  = tf_a.dim_size(0);
      H   = tf_a.dim_size(1);
      W   = tf_a.dim_size(2);
      B   = tf_a.dim_size(3);
      // shapes for b
      int64_t Cb  = tf_b.dim_size(0);
      F   = tf_b.dim_size(1);
      // checks
      OP_REQUIRES(context, Ca == Cb, tensorflow::errors::InvalidArgument("operands must have the same number of channels"));
      C = Ca;
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
    // shift configuration
    int32_t* shift_h_data = h_shift_h_.flat<int32_t>().data();
    int32_t* shift_w_data = h_shift_w_.flat<int32_t>().data();
    std::vector<int32_t> shift_h(shift_h_data, shift_h_data + C);
    std::vector<int32_t> shift_w(shift_w_data, shift_w_data + C);
    shift_key_t key = {B, C, 1, H, W, 1, R_, S_, F, shift_h_data, shift_w_data, OP, has_bias};
    // create configuration
    triton::dnn::shift* shift;
    if(m_config.find(key) == m_config.end())
      shift = m_config.emplace(key, new triton::dnn::shift(
                                                  B, C, D, H, W, T, R_, S_, F,
                                                  shift_h, shift_w, "fp32", "fp32", OP, has_bias))
                                                    .first->second.get();
    else
      shift = m_config.at(key).get();

    // shapes for c
    std::vector<int64> c_shapes;
    for(int32_t x: shift->c_shapes())
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
    // get JIT
    triton::jit* jit;
    bool autotune = false;
    if(m_jit.find(key) == m_jit.end()) {
      jit = m_jit.emplace(key, new triton::jit(ctx)).first->second.get();
      std::ostringstream oss;
      shift->src(oss);
      std::string src = oss.str();
      auto benchmark = [&](triton::driver::kernel* kernel,
                           triton::jit::launch_information info) {
        // launch info
        unsigned TM = info.global_range_size[0];
        unsigned TN = info.global_range_size[1];
        unsigned nthreads = info.num_threads;
        shift->init(stream, (triton::driver::cu_module*)kernel->module());
        shift->enqueue(stream, kernel, &da, &db, &dc, TM, TN, nthreads);
        stream->synchronize();
        double ts = triton::tools::bench([&](){ shift->enqueue(stream, kernel, &da, &db, &dc, TM, TN, nthreads); },
                          [&](){ stream->synchronize(); }, ctx->device());
        return  shift->get_nflops() / ts * 1e-3;
      };
      // auto-tune and save result
      if(autotune) {
        triton::jit::tune_res_t best = jit->autotune("shift", src.c_str(), benchmark);
        jit->add_module("shift", src.c_str(), best.params);
      }
      else {
        jit->add_module("shift", src.c_str(), jit->get_valid("shift", src.c_str()));
      }
      triton::driver::kernel* kernel = jit->get_function("shift");
      shift->init(stream, (triton::driver::cu_module*)kernel->module());
    }
    else
      jit = m_jit.at(key).get();
    // Run
    triton::driver::kernel* kernel = jit->get_function("shift");
    triton::jit::launch_information info = jit->get_launch_info("shift");
    // launch info
    unsigned TM = info.global_range_size[0];
    unsigned TN = info.global_range_size[1];
    unsigned nthreads = info.num_threads;
    // enqueue
    shift->enqueue(stream, kernel, &da, &db, &dc, TM, TN, nthreads);
  }

private:
  Tensor  h_shift_h_;
  Tensor  h_shift_w_;
  int R_;
  int S_;
};

REGISTER_KERNEL_BUILDER(Name("ShiftConv").Device(DEVICE_GPU), ShiftConvOp<triton::dnn::shift::FPROP>);
REGISTER_OP("ShiftConv")
    .Input("a: float32")
    .Input("b: float32")
    .Attr("shift_h: tensor")
    .Attr("shift_w: tensor")
    .Output("c: float32");

REGISTER_KERNEL_BUILDER(Name("ShiftConvDx").Device(DEVICE_GPU), ShiftConvOp<triton::dnn::shift::BPROP>);
REGISTER_OP("ShiftConvDx")
    .Input("a: float32")
    .Input("b: float32")
    .Attr("shift_h: tensor")
    .Attr("shift_w: tensor")
    .Output("c: float32");

REGISTER_KERNEL_BUILDER(Name("ShiftConvDw").Device(DEVICE_GPU), ShiftConvOp<triton::dnn::shift::WGRAD>);
REGISTER_OP("ShiftConvDw")
    .Input("a: float32")
    .Input("b: float32")
    .Attr("shift_h: tensor")
    .Attr("shift_w: tensor")
    .Output("c: float32");

