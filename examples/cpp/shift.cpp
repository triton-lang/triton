#include <cstring>
#include <cstdio>
#include <sstream>
#include "triton/runtime/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/tools/bench.hpp"
#include "triton/dnn/shift.h"
#include "triton/external/half.hpp"

int main() {
  typedef half_float::half NumericT;
  std::string numeric_t_str = "fp16";

  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();

  // initialization
  int32_t R = 3, S = 3;
  int32_t BS = 4, F = 1024;
  int32_t H = 16, W = 16;
  int32_t C = 1024;

  // random shifts
  std::vector<int32_t> shift_h(C);
  std::vector<int32_t> shift_w(C);
  for(int32_t c = 0; c < C; c++){
    shift_h[c] = rand() % R - R/2;
    shift_w[c] = rand() % S - S/2;
  }
  // configuration
  triton::dnn::shift shift(BS, C, 1, H, W, 1, R, S, F, 1, 1, shift_h, shift_w, numeric_t_str, numeric_t_str, triton::dnn::shift::BPROP);
  // host buffers
  std::vector<float> hc(shift.c_size());
  std::vector<float> rc(shift.c_size());
  std::vector<NumericT> ha(shift.a_size());
  std::vector<NumericT> hb(shift.b_size());
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
  shift.enqueue(stream, da, db, dc);
//  stream->read(dc, true, 0, hc);
//  shift.cpu_ref(rc.data(), ha.data(), hb.data());
//  for(size_t i = 0; i < hc.size(); i++)
//    if(std::isnan(hc[i]) || std::abs(hc[i] - rc[i])/std::max(hc[i], rc[i]) > 1e-4){
//      std::cout << i << " " << hc[i] << " " << rc[i] << std::endl;
//      exit(EXIT_FAILURE);
//    }
//  std::cout << "Pass!" << std::endl;

}
