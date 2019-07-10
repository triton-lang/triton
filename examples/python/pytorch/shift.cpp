#include <vector>
#include <sstream>
#include <torch/torch.h>
#include <torch/script.h>
#include "ATen/cuda/CUDAContext.h"
#include "triton/runtime/jit.h"
#include "triton/driver/stream.h"
#include "triton/dnn/shift.h"
#include "triton/tools/bench.hpp"

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

typedef std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t,
                   int32_t, int32_t, int32_t, int32_t,
                   int32_t*, int32_t*,
                   triton::dnn::shift::type, bool> shift_key_t;

static std::map<CUstream, std::unique_ptr<triton::driver::stream>>  m_shift_stream;
static std::map<shift_key_t,  std::unique_ptr<triton::jit>>         m_shift_jit;
static std::map<shift_key_t, std::unique_ptr<triton::dnn::shift>>   m_shift_config;

torch::Tensor shift_common(
    int32_t B, int32_t C, int32_t D, int32_t H, int32_t W,
    int32_t T, int32_t R, int32_t S, int32_t F,
    std::vector<int32_t> shift_h, std::vector<int32_t> shift_w,
    triton::dnn::shift::type ty,
    torch::Tensor torcha, torch::Tensor torchb, torch::Tensor torchbias,
    bool autotune = false
    ) {

  // Wrap CUDA handles
  c10::DeviceIndex device = torcha.storage().device().index();

  // Get stream
  CUstream custream = (CUstream)at::cuda::getCurrentCUDAStream(device).stream();
  triton::driver::stream* stream;
  if(m_shift_stream.find(custream) == m_shift_stream.end())
    stream = m_shift_stream.emplace(custream, new triton::driver::cu_stream(custream, false)).first->second.get();
  else
    stream = m_shift_stream.at(custream).get();

  // Get context
  triton::driver::context* ctx = stream->context();

  // Get configuration
  bool has_bias = torchbias.storage().size() > 0;
  shift_key_t key = {B, C, D, H, W, T, R, S, F, shift_h.data(), shift_w.data(), ty, has_bias};
  triton::dnn::shift* configuration;
  if(m_shift_config.find(key) == m_shift_config.end())
    configuration = m_shift_config.emplace(key, new triton::dnn::shift(
                                                B, C, D, H, W, T, R, S, F,
                                                shift_h, shift_w, "fp32", "fp32",
                                                ty, has_bias)).first->second.get();
  else
    configuration = m_shift_config.at(key).get();

  // Bind memory
  triton::driver::cu_buffer a(ctx, (CUdeviceptr)torcha.storage().data(), false);
  triton::driver::cu_buffer b(ctx, (CUdeviceptr)torchb.storage().data(), false);
  triton::driver::cu_buffer cubias(ctx, (CUdeviceptr)torchbias.storage().data(), false);
  triton::driver::buffer* bias = has_bias ? &cubias : nullptr;

  // Allocate output
  std::vector<int32_t> c_shapes = configuration->c_shapes();
  torch::Tensor torchc = torch::empty({c_shapes[0], c_shapes[1], c_shapes[2], c_shapes[3]}).cuda();
  triton::driver::cu_buffer c(ctx, (CUdeviceptr)torchc.storage().data(), false);

  // Get JIT
  triton::jit* jit;
  if(m_shift_jit.find(key) == m_shift_jit.end()){
    jit = m_shift_jit.emplace(key, new triton::jit(ctx)).first->second.get();
    std::ostringstream oss;
    configuration->triton_c_src(oss);
    std::string src = oss.str();
    // benchmark a given shiftolution kernel
    auto benchmark = [&](triton::driver::kernel* kernel,
                         triton::jit::launch_information info) {
      configuration->init_impl(stream, (triton::driver::cu_module*)kernel->module());
      unsigned TM = info.global_range_size[0];
      unsigned TN = info.global_range_size[1];
      unsigned nthreads = info.num_threads;
      configuration->enqueue_impl(stream, kernel, &a, &b, &c, TM, TN, nthreads);
      stream->synchronize();
      double ts = triton::tools::bench([&](){ configuration->enqueue_impl(stream, kernel, &a, &b, &c, TM, TN, nthreads); },
                        [&](){ stream->synchronize(); }, stream->context()->device());
      return configuration->num_flops() / ts * 1e-3;
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
    configuration->init_impl(stream, (triton::driver::cu_module*)kernel->module());
  }
  else
    jit = m_shift_jit.at(key).get();

  // Run
  triton::driver::kernel* kernel = jit->get_function("shift");
  triton::jit::launch_information info = jit->get_launch_info("shift");
  // launch info
  unsigned TM = info.global_range_size[0];
  unsigned TN = info.global_range_size[1];
  unsigned nthreads = info.num_threads;
  // enqueue
  configuration->enqueue_impl(stream, kernel, &a, &b, &c, TM, TN, nthreads);
  return torchc;
}
