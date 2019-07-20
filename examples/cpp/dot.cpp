#include <cstring>
#include <cstdio>
#include "triton/runtime/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/dnn/gemm.h"
#include "triton/tools/bench.hpp"


int main() {
  bool AT = false;
  bool BT = true;
  typedef float T;
  std::string ty = "fp16";
  size_t dt_nbytes = sizeof(T);
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  // matrix multiplication parameters
  int32_t M = 4096, N = 4096, K = 4096;
  std::vector<T> hc(M*N);
  std::vector<T> rc(M*N);
  std::vector<T> ha(M*K);
  std::vector<T> hb(K*N);
  srand(0);
  for(size_t i = 0; i < ha.size(); i++)
    ha[i] = (T)rand()/RAND_MAX;
  for(size_t i = 0; i < hb.size(); i++)
    hb[i] = (T)rand()/RAND_MAX;
  for(size_t i = 0; i < hc.size(); i++)
    hc[i] = 0;
  triton::driver::buffer* dc = triton::driver::buffer::create(context, hc.size()*dt_nbytes);
  triton::driver::buffer* da = triton::driver::buffer::create(context, ha.size()*dt_nbytes);
  triton::driver::buffer* db = triton::driver::buffer::create(context, hb.size()*dt_nbytes);
  triton::driver::stream* stream = triton::driver::stream::create(context);
  stream->write(da, true, 0, ha);
  stream->write(db, true, 0, hb);
  stream->write(dc, true, 0, hc);
  stream->synchronize();
  triton::dnn::gemm gemm(M, N, K, AT, BT, ty, ty, 4, 4);
  gemm.enqueue(stream, {da, db, dc}, true);
//  stream->read(dc, true, 0, hc);
//  gemm.cpu_ref<T>(rc, ha, hb);
//  for(size_t i = 0; i < M*N; i++)
//    if(std::isnan(hc[i]) || std::abs(hc[i] - rc[i])/std::max(hc[i], rc[i]) > 1e-4){
//      std::cout << i << " " << hc[i] << " " << rc[i] << std::endl;
//      exit(EXIT_FAILURE);
//    }
//  std::cout << "Pass!" << std::endl;
}
