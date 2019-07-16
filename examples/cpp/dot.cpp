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
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  // matrix multiplication parameters
  int32_t M = 131072, N = 128, K = 128;
  std::vector<float> hc(M*N);
  std::vector<float> rc(M*N);
  std::vector<float> ha(M*K);
  std::vector<float> hb(K*N);
  srand(0);
  for(size_t i = 0; i < ha.size(); i++)
    ha[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < hb.size(); i++)
    hb[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < hc.size(); i++)
    hc[i] = 0;
  triton::driver::buffer* dc = triton::driver::buffer::create(context, hc.size()*4);
  triton::driver::buffer* da = triton::driver::buffer::create(context, ha.size()*4);
  triton::driver::buffer* db = triton::driver::buffer::create(context, hb.size()*4);
  triton::driver::stream* stream = triton::driver::stream::create(context);
  stream->write(da, true, 0, ha);
  stream->write(db, true, 0, hb);
  stream->write(dc, true, 0, hc);
  stream->synchronize();
  triton::dnn::gemm gemm(M, N, K, AT, BT, "fp32", "fp32", 4, 4);
  gemm.enqueue(stream, {da, db, dc});
  stream->read(dc, true, 0, hc);
  gemm.cpu_ref<float>(rc, ha, hb);
  for(size_t i = 0; i < M*N; i++)
    if(!std::isnan(hc[i]) && std::abs(hc[i] - rc[i])/std::max(hc[i], rc[i]) > 1e-4){
      std::cout << i << " " << hc[i] << " " << rc[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  std::cout << "Pass!" << std::endl;
}
