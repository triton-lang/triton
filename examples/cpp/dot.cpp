#include <cstring>
#include <sstream>
#include <cstdio>
#include "triton/runtime/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/dnn/gemm.h"
#include "triton/tools/bench.hpp"
#include "cuda.h"

template<class T>
void diff(const std::vector<T>& x, const std::vector<T>& y){
    for(size_t i = 0; i < x.size(); i++)
      if(std::isnan(x[i]) || std::abs(x[i] - y[i])/std::max(x[i], y[i]) > 1e-4){
        std::cout << i << " " << x[i] << " " << y[i] << std::endl;
        exit(EXIT_FAILURE);
      }
    std::cout << "Pass!" << std::endl;
}

struct perf_t {
  double triton;
  double cublas;
};


perf_t do_bench(triton::driver::stream* stream, bool AT, bool BT, int32_t M, int32_t N, int32_t K){
  typedef float NumericT;
  std::string ty = "fp16";
  size_t dt_nbytes = sizeof(NumericT);
  triton::driver::context* context = stream->context();
  std::vector<NumericT> hc(M*N);
  std::vector<NumericT> ha(M*K);
  std::vector<NumericT> hb(K*N);
  srand(0);
  for(size_t i = 0; i < ha.size(); i++)
    ha[i] = (NumericT)rand()/RAND_MAX;
  for(size_t i = 0; i < hb.size(); i++)
    hb[i] = (NumericT)rand()/RAND_MAX;
  for(size_t i = 0; i < hc.size(); i++)
    hc[i] = 0;
  triton::driver::buffer* dc = triton::driver::buffer::create(context, hc.size()*dt_nbytes);
  triton::driver::buffer* da = triton::driver::buffer::create(context, ha.size()*dt_nbytes);
  triton::driver::buffer* db = triton::driver::buffer::create(context, hb.size()*dt_nbytes);
  stream->write(da, true, 0, ha);
  stream->write(db, true, 0, hb);
  stream->write(dc, true, 0, hc);
  stream->synchronize();
  triton::dnn::dot dot(M, N, K, AT, BT, ty, ty, 8, 8);
  // benchmark triton
  double triton_ns = triton::tools::bench([&]() { dot.enqueue(stream, {da, db, dc}, triton::dnn::PARTIAL_TUNING);}, stream);
  // benchmark cublas
  NumericT alpha = 1;
  NumericT beta = 0;
  int32_t lda = AT ? K : M;
  int32_t ldb = BT ? N : K;
  int32_t ldc = M;
  cublasGemmAlgo_t fastest;
//  cublasGemm(HALF_TYPE, stream, AT, BT, M, N, K,
//             &alpha, da, lda,
//             db, ldb, &beta,
//             dc, ldc, &fastest);
  double cublas_ns = triton::tools::bench([&]() {   cublasGemm(HALF_TYPE, stream, AT, BT, M, N, K,
                                                               &alpha, da, lda,
                                                               db, ldb, &beta,
                                                               dc, ldc, nullptr, CUBLAS_GEMM_DEFAULT_TENSOR_OP); }, stream);
  // result
  auto tflops = [&](double nanosec) { return dot.num_flops() / nanosec * 1e-3; };

  perf_t result;
  result.cublas = tflops(cublas_ns);
  result.triton = tflops(triton_ns);
  // clean-up
  delete dc;
  delete da;
  delete db;
  return result;
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

    perf_t perf(triton::driver::stream *stream){
      return do_bench(stream, AT, BT, M, N, K);
    }
  };
  // shapes to benchmark
  std::vector<config_t> configs = {
//    {false, false, 8192, 512, 512},
    {false, true, 8192, 8192, 8192},
    {false, true, 32768, 256, 512}
//    {true,  false, 8192, 512, 512},
//    {true,  true,  8192, 512, 512}
  };
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context);
  // does the work
  for(config_t c: configs){
    perf_t perf = c.perf(stream);
    std::cout << c.repr() << ", " << perf.triton << ", " << perf.cublas << std::endl;
  }
}
