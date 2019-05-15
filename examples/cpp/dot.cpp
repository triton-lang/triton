#include <cstring>
#include <cstdio>
#include "common.hpp"
#include "triton/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/dnn/gemm.h"


int main() {
  bool AT = false;
  bool BT = true;

  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::jit jit(context);

  // matrix multiplication parameters
  int32_t M = 512, N = 512, K = 512;
  std::vector<float> hc(M*N);
  std::vector<float> rc(M*N);
  std::vector<float> ha(M*K);
  std::vector<float> hb(K*N);
  std::vector<int32_t> hlocks(2048);
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
  triton::driver::buffer* dlocks = triton::driver::buffer::create(context, hlocks.size()*4);
  triton::driver::stream* stream = triton::driver::stream::create(context);
  stream->write(da, true, 0, ha);
  stream->write(db, true, 0, hb);
  stream->write(dc, true, 0, hc);
  triton::dnn::gemm::init(stream, dlocks);
  stream->synchronize();


  // benchmark a given matrix multiplication kernel
  auto benchmark = [&](triton::driver::kernel* kernel,
                       triton::jit::launch_information info) {
    unsigned TM = info.global_range_size[0];
    unsigned TN = info.global_range_size[1];
    unsigned nthreads = info.num_threads;
    unsigned GZ = jit.get_int("GZ");
    std::array<size_t, 3> grid = {(M + TM - 1)/TM, (N + TN - 1)/TN, GZ};
    triton::dnn::gemm::set_arg(kernel, da, db, dc, M, N, K, dlocks, grid[0], grid[1]);
    stream->enqueue(kernel, grid, {nthreads, 1, 1});
    stream->synchronize();
    double ts = bench([&](){stream->enqueue(kernel, grid, {nthreads, 1, 1});},
                      [&](){ stream->synchronize(); }, *context->device());
    return  2.*M*N*K / ts * 1e-3;
  };


  // just-in-time compile source-code
  std::string src = triton::dnn::gemm::src(AT, BT);
//  jit.autotune("matmul",src.c_str(), benchmark);
  jit.add_module("matmul", src.c_str(), triton::dnn::gemm::default_params(AT, BT));
  triton::driver::kernel* kernel = jit.get_function("matmul");
  triton::jit::launch_information info = jit.get_launch_info("matmul");
  std::cout << "Performance: " << benchmark(kernel, info) << " TFLOPS " << std::endl;
  stream->read(dc, true, 0, hc);
  simple_gemm<float>(AT, BT, rc, ha, hb, M, N, K);
  for(size_t i = 0; i < M*N; i++)
    if(!std::isnan(hc[i]) && std::abs(hc[i] - rc[i])/std::max(hc[i], rc[i]) > 1e-4){
      std::cout << i << " " << hc[i] << " " << rc[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  std::cout << "Pass!" << std::endl;
}
