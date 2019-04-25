#include <cstring>
#include <cstdio>
#include "common.hpp"
#include "triton/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"

const char* src =
R"(
const tunable int32 TM = {16, 32, 64, 128};
const tunable int32 TN = {16, 32, 64, 128};
const tunable int32 TK = {8};
const tunable int32 GZ = {1};

void matmul(restrict read_only fp32 *A, restrict read_only fp32 *B, fp32 *C,
           int32 M, int32 N, int32 K,
           int32 lda, int32 ldb, int32 ldc,
           int32 *locks, int32 grid0, int32 grid1) {
  int32 rxa[TM] = get_global_range[TM](0);
  int32 ryb[TN] = get_global_range[TN](1);
  int32 rz = get_global_range[1](2);
  int32 rka[TK] = 0 ... TK;
  int32 rkb[TK] = 0 ... TK;
  fp32 c[TM, TN] = 0;
  int32 div = K / GZ;
  int32 rem = K % GZ;
  K = select(rz < rem, div - 1, div);
  int32 offk = select(rz < rem, rz*(div + 1), rz*div + rem);
  fp32* pa[TM, TK] = A + (offk + rka[newaxis, :])*lda + rxa[:, newaxis];
  fp32* pb[TN, TK] = B + (offk + rkb[newaxis, :])*ldb + ryb[:, newaxis];
  fp32 a[TM, TK] = *pa;
  fp32 b[TN, TK] = *pb;
  int32 last_a = ((M*K - 1) - (TM*TK + 1)) / lda;
  int32 last_b = ((K*N - 1) - (TN*TK + 1)) / ldb;
  last_a = last_a / TK * TK;
  last_b = last_b / TK * TK;
  int32 bound = K - max(last_a, last_b);
  for(int32 k = K; k > bound; k = k - TK){
    c = dot(a, trans(b), c);
    pa = pa + TK*lda;
    pb = pb + TK*ldb;
    a = *pa;
    b = *pb;
  }
  int32 rxc[TM] = get_global_range[TM](0);
  int32 ryc[TN] = get_global_range[TN](1);
  for(int32 k = bound; k > 0; k = k - 1){
    int1 checka[TM, 1] = rxc[:, newaxis] < M;
    int1 checkb[TN, 1] = ryc[:, newaxis] < N;
    fp32* pa[TM, 1] = A + (offk + K - k)*lda + rxc[:, newaxis];
    fp32* pb[TN, 1] = B + (offk + K - k)*ldb + ryc[:, newaxis];
    fp32 a[TM, 1] = checka ? *pa : 0;
    fp32 b[TN, 1] = checkb ? *pb : 0;
    c = dot(a, trans(b), c);
  }
  int32 ridx = get_range_id(0);
  int32 ridy = get_range_id(1);
  fp32* pc[TM, TN] = C + ryc[newaxis, :]*ldc + rxc[:, newaxis];
  int32 *plock = locks + ridx + ridy*grid0;
  for(int32 L =  __atomic_cas(plock, 0, 1); L == 1; L = __atomic_cas(plock, 0, 1)){}
  int32 *pcount = plock + grid0*grid1;
  int32 count = *pcount;
  int32 countp1 = select(count == GZ - 1, 0, count + 1);
  int1 checkc0[TM] = rxc < M;
  int1 checkc1[TN] = ryc < N;
  int1 checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  if(count == 0) {
    @checkc *pc = c;
    *pcount = countp1;
  }
  else {
    @checkc *pc = c + (checkc ? *pc : 0);
    *pcount = countp1;
  }
  __atomic_cas(plock, 1, 0);
}
)";

int main() {
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
  stream->synchronize();


  // benchmark a given matrix multiplication kernel
  auto benchmark = [&](triton::driver::kernel* kernel,
                       triton::jit::launch_information info) {
    // launch info
    unsigned TM = info.global_range_size[0];
    unsigned TN = info.global_range_size[1];
    unsigned nthreads = info.num_threads;
    unsigned GZ = jit.get_int("GZ");
    std::array<size_t, 3> grid = {(M + TM - 1)/TM, (N + TN - 1)/TN, GZ};
    // init locks
    stream->write(dlocks, true, 0, hlocks);
    // set argument
    kernel->setArg(0, da);
    kernel->setArg(1, db);
    kernel->setArg(2, dc);
    kernel->setArg(3, M);
    kernel->setArg(4, N);
    kernel->setArg(5, K);
    kernel->setArg(6, M);
    kernel->setArg(7, N);
    kernel->setArg(8, M);
    kernel->setArg(9, dlocks);
    kernel->setArg(10, grid[0]);
    kernel->setArg(11, grid[1]);
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
    16, 2, 64, 16, 2, 64, 16, 8, 2, 2, 8, 8, 8, 1
  };
//  jit.autotune("matmul",src, benchmark);
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
