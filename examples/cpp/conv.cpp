#include <cstring>
#include <cstdio>
#include <sstream>
#include "triton/runtime/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/dnn/conv.h"
#include "triton/tools/bench.hpp"

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::dnn::conv::type ty = triton::dnn::conv::FPROP;
  // initialization
  int32_t B = 16, NF = 128;
  int32_t D = 1, H = 16, W = 16;
  int32_t NC = 64, T = 1, R = 3, S = 3;
  int32_t pad_d = 0, pad_h = 0, pad_w = 0;
  int32_t stride_d = 1, stride_h = 1, stride_w = 1;
  int32_t upsample_d = 1, upsample_h = 1, upsample_w = 1;
//  triton::dnn::conv configuration(128, 256, 1, 14, 14, 1, 5, 5, 512, 1, 1, 1, 0, 0, 0, 1, 1, 1, "float", "float", triton::dnn::conv::FPROP, 0);
  triton::dnn::conv configuration(B, NC, D, H, W, T, R, S, NF,
                                  stride_d, stride_h, stride_w,
                                  pad_d, pad_h, pad_w,
                                  upsample_d, upsample_h, upsample_w,
                                  "float", "float", ty, 0);
  // convolution configuration
  std::vector<float> hc(configuration.c_size());
  std::vector<float> rc(configuration.c_size());
  std::vector<float> ha(configuration.a_size());
  std::vector<float> hb(configuration.b_size());
  srand(0);
  for(size_t i = 0; i < ha.size(); i++)
    ha[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < hb.size(); i++)
    hb[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < hc.size(); i++)
    hc[i] = 0;
  rc = hc;
  triton::driver::buffer* dc = triton::driver::buffer::create(context, hc.size()*4);
  triton::driver::buffer* da = triton::driver::buffer::create(context, ha.size()*4);
  triton::driver::buffer* db = triton::driver::buffer::create(context, hb.size()*4);
  triton::driver::stream* stream = triton::driver::stream::create(context);
  stream->write(da, true, 0, ha);
  stream->write(db, true, 0, hb);
  stream->write(dc, true, 0, hc);
  stream->synchronize();
  configuration.enqueue(stream, {da, db, dc, nullptr});
  stream->read(dc, true, 0, hc);
  configuration.cpu_ref(rc.data(), ha.data(), hb.data());
  for(size_t i = 0; i < hc.size(); i++){
    if(std::isnan(hc[i]) || std::abs(hc[i] - rc[i])/std::max(hc[i], rc[i]) > 1e-4){
      std::cout << i << " " << hc[i] << " " << rc[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::cout << "Pass!" << std::endl;
}
