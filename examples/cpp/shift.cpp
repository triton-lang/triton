#include <cstring>
#include <cstdio>
#include <sstream>
#include "triton/runtime/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/tools/bench.hpp"
#include "triton/dnn/shift.h"
#include "triton/external/half.hpp"

double do_bench(triton::driver::context* context,
                int32_t R, int32_t S, int32_t B, int32_t F, int32_t H, int32_t W, int32_t C,
                triton::dnn::op_t op, triton::dnn::layout_t layout,
                std::string numeric_t) {
  typedef float NumericT;

  // random shifts
  std::vector<int32_t> shift_h(C);
  std::vector<int32_t> shift_w(C);
  for(int32_t c = 0; c < C; c++){
    shift_h[c] = rand() % R - R / 2;
    shift_w[c] = rand() % S - S / 2;
  }
  // configuration
  triton::dnn::shift shift(B, C, 1, H, W, 1, R, S, F, 1, 1,
                           shift_h.data(), shift_w.data(),
                           numeric_t, numeric_t,
                           op, false, layout);
  // host buffers
  size_t a_size = B*C*H*W;
  size_t b_size = C*F;
  size_t c_size = B*F*H*W;
  if(op == triton::dnn::BPROP)
    std::swap(a_size, c_size);
  if(op == triton::dnn::WGRAD){
    std::swap(b_size, c_size);
    std::swap(a_size, b_size);
  }
  std::vector<NumericT> ha(a_size);
  std::vector<NumericT> hb(b_size);
  std::vector<float> hc(c_size);
  std::vector<float> rc(hc.size());
  // device buffers
  triton::driver::buffer* dc = triton::driver::buffer::create(context, hc.size()*4);
  triton::driver::buffer* da = triton::driver::buffer::create(context, ha.size()*sizeof(NumericT));
  triton::driver::buffer* db = triton::driver::buffer::create(context, hb.size()*sizeof(NumericT));
  triton::driver::stream* stream = triton::driver::stream::create(context);
  // initialize host
  srand(0);
  for(size_t i = 0; i < ha.size(); i++)
    ha[i] = (NumericT)rand() / RAND_MAX;
  for(size_t i = 0; i < hb.size(); i++)
    hb[i] = (NumericT)rand() / RAND_MAX;
  for(size_t i = 0; i < hc.size(); i++)
    hc[i] = 0;
  // initialize device
  stream->write(da, true, 0, ha);
  stream->write(db, true, 0, hb);
  stream->write(dc, true, 0, hc);
  stream->synchronize();
  double nanosec = triton::tools::bench([&]() { shift.enqueue(stream, {da, db, dc});}, stream);
  return shift.num_flops() / nanosec * 1e-3;
}

int main() {
  using triton::dnn::op_t;
  using triton::dnn::layout_t;

  struct config_t{
    int32_t B;
    int32_t C;
    int32_t H;
    int32_t W;
    int32_t R;
    int32_t S;
    int32_t F;
    int32_t stride_h;
    int32_t stride_w;
    op_t op;
    layout_t layout;
    std::string ty;

    std::string repr() {
      std::ostringstream oss;
      oss << B << ", " << C << ", " << H << ", " << W << ", " << R << ", " << S << ", " << F << ", " << op << ", " << layout << ", " << ty;
      return oss.str();
    }

    double perf(triton::driver::context *context){
      return do_bench(context, R, S, B, F, H, W, C, op, layout, ty);
    }
  };
  // shapes to benchmark
  std::vector<config_t> configs;
  std::vector<config_t> resnet18 = {
    {128, 128, 32, 32, 3, 3, 128, 1, 1},
    {128, 128, 32, 32, 3, 3, 256, 2, 2},
    {128, 256, 16, 16, 3, 3, 256, 1, 1},
    {128, 256, 16, 16, 3, 3, 512, 2, 2},
    {128, 512,  8,  8, 3, 3, 512, 1, 1},
    {128, 512,  8,  8, 3, 3, 1024, 1, 1},
    {128, 1024, 8, 8, 3, 3, 1024, 1, 1}
  };
  for(config_t c: resnet18){
    for(op_t op: {op_t::FPROP, op_t::BPROP, op_t::WGRAD})
      configs.push_back({c.B, c.C, c.H, c.W, c.R, c.S, c.F, c.stride_h, c.stride_w, op, layout_t::CHWN, "fp16"});
  }

  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  for(config_t c: configs)
    std::cout << c.repr() << ", " << c.perf(context) << std::endl;

}
