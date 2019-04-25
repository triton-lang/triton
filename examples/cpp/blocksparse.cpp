#include <cstring>
#include <cstdio>
#include "common.hpp"
#include "triton/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"

const char* src =
R"(
const tunable int32 TM = {16, 32, 64, 128};
const tunable int32 TN = {8};
const tunable int32 TK = {8};

void blocksparse(restrict read_only fp32 *a, restrict read_only fp32 *b, fp32 *c,
                int32 M, int32 N, int32 K, int32 bound){
  int32 rxa[TM] = get_global_range[TM](0);
  int32 ryb[TN] = get_global_range[TN](1);
  int32 rka[TK] = 0 ... TK;
  int32 rkb[TK] = 0 ... TK;
  fp32 C[TM, TN] = 0;
  fp32* pa[TM, TK] = a + rka[newaxis, :]*M + rxa[:, newaxis];
  fp32* pb[TN, TK] = b + rkb[newaxis, :]*K + ryb[:, newaxis];
  fp32 a[TM, TK] = *pa;
  fp32 b[TN, TK] = *pb;
  for(int32 k = K; k > 0;){
    C = dot(a, trans(b), C);
    pa = pa + TK*M;
    pb = pb + TK*N;
    k = k - TK;
    int1 checka[TM, TK] = k > bound;
    int1 checkb[TN, TK] = k > bound;
    @checka a = *pa;
    @checkb b = *pb;
    if(k > bound)
      continue;
    int1 checka0[TM] = rxa < M;
    int1 checka1[TK] = rka < k;
    int1 checkb0[TN] = ryb < N;
    int1 checkb1[TK] = rkb < k;
    checka = checka0[:, newaxis] && checka1[newaxis, :];
    checkb = checkb0[:, newaxis] && checkb1[newaxis, :];
    a = checka ? *pa : 0;
    b = checkb ? *pb : 0;
  }
  int32 rxc[TM] = get_global_range[TM](0);
  int32 ryc[TN] = get_global_range[TN](1);
  fp32* pc[TM, TN] = c + ryc[newaxis, :]*M + rxc[:, newaxis];
  int1 checkc0[TM] = rxc < M;
  int1 checkc1[TN] = ryc < N;
  int1 checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  @checkc *pc = C;
}
)";

std::vector<int> make_deltas(std::vector<int> mask, int K, int N){
  std::vector<std::vector<std::pair<int,int>>> pairs(N);
  unsigned int current = 0;
  for(int k = 0; k < K; k++)
  for(int n = 0; n < N; n++){
    if(mask[k + n*K])
      pairs[n].push_back({current, k});
  }
}

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::jit jit(context);


  // matrix multiplication parameters
  int32_t M = 512, N = 32, K = 2048;
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


  // benchmark a given matrix multiplication kernel
  auto benchmark = [&](triton::driver::kernel* kernel,
                       triton::jit::launch_information info) {
    // launch info
    unsigned TM = info.global_range_size[0];
    unsigned TN = info.global_range_size[1];
    unsigned nthreads = info.num_threads;
    std::array<size_t, 3> grid = {(M + TM - 1)/TM, (N + TN - 1)/TN, 1};
    // fast bounds-checking
    unsigned TK = jit.get_int("TK");
    unsigned lasti = (grid[0]*TM - 1)*TM + TM - 1;
    unsigned lastj = (grid[1]*TN - 1)*TN + TN - 1;
    unsigned lastk = TK - 1;
    bool AT = false;
    bool BT = true;
    unsigned last_safe_a = (AT==false)?(M*K - 1 - lasti)/M - lastk : M*K - 1 - lasti*K - lastk;
    unsigned last_safe_b =  (BT==true)?(N*K - 1 - lastj)/N - lastk : N*K - 1 - lastj*K - lastk;
    int32_t bound = std::max<unsigned>(1, std::max(K - last_safe_a, K - last_safe_b));
    // set argument
    kernel->setArg(0, da);
    kernel->setArg(1, db);
    kernel->setArg(2, dc);
    kernel->setArg(3, M);
    kernel->setArg(4, N);
    kernel->setArg(5, K);
    kernel->setArg(6, bound);
    // dry run
    stream->enqueue(kernel, grid, {nthreads, 1, 1});
    stream->synchronize();
    // benchmark
    double ts = bench([&](){stream->enqueue(kernel, grid, {nthreads, 1, 1});},
                      [&](){ stream->synchronize(); }, *context->device());
    ts = ts * 1e-9;
    double tflops = 2.*M*N*K / ts * 1e-12;
    return tflops;
  };


  // just-in-time compile source-code
  std::vector<unsigned> params = {
    16, 2, 64,
    32, 2, 64,
    16, 8, 2, 2,
    8, 8,
    4
  };
  jit.autotune("matmul",src, benchmark);
  jit.add_module("matmul", src, params);
  triton::driver::kernel* kernel = jit.get_function("matmul");
  triton::jit::launch_information info = jit.get_launch_info("matmul");
  std::cout << "Performance: " << benchmark(kernel, info) << " TFLOPS " << std::endl;
  stream->read(dc, true, 0, hc);
  simple_gemm<float,false,true>(rc, ha, hb, M, N, K);
  for(size_t i = 0; i < M*N; i++)
    if(std::abs(hc[i] - rc[i])/std::max(hc[i], rc[i]) > 1e-4){
      std::cout << i << " " << hc[i] << " " << rc[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  std::cout << "Pass!" << std::endl;
}
