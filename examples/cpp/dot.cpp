#include <cstring>
#include <sstream>
#include <cstdio>
#include "triton/runtime/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/dnn/gemm.h"
#include "triton/tools/bench.hpp"

template<class T>
void diff(const std::vector<T>& x, const std::vector<T>& y){
    for(size_t i = 0; i < x.size(); i++)
      if(std::isnan(x[i]) || std::abs(x[i] - y[i])/std::max(x[i], y[i]) > 1e-4){
        std::cout << i << " " << x[i] << " " << y[i] << std::endl;
        exit(EXIT_FAILURE);
      }
    std::cout << "Pass!" << std::endl;
}

double do_bench(triton::driver::context* context, bool AT, bool BT, int32_t M, int32_t N, int32_t K){
  typedef float T;
  std::string ty = "fp16";
  size_t dt_nbytes = sizeof(T);
  std::vector<T> hc(M*N);
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
  triton::dnn::dot dot(M, N, K, AT, BT, ty, ty, 8, 8);
  double nanosec = triton::tools::bench([&]() { dot.enqueue(stream, {da, db, dc}, triton::dnn::PARTIAL_TUNING);}, stream);
  delete dc;
  delete da;
  delete db;
  return dot.num_flops() / nanosec * 1e-3;
}

int main() {
  struct config_t{
    bool AT;
    bool BT;
    int32_t M;
    int32_t N;
    int32_t K;

    std::string repr() {
      std::ostringstream oss;
      oss << AT << " " << BT << " " << M << " " << N << " " << K;
      return oss.str();
    }

    double perf(triton::driver::context *context){
      return do_bench(context, AT, BT, M, N, K);
    }
  };
  // shapes to benchmark
  std::vector<config_t> configs = {
    {false, false, 4096, 4096, 4096},
    {false, true,  4096, 4096, 4096},
    {true,  false, 4096, 4096, 4096},
    {true,  true,  4096, 4096, 4096}
  };
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  // does the work
  for(config_t c: configs){
    std::cout << c.repr() << ", " << c.perf(context) << std::endl;
  }
}
