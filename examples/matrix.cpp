#include <cstring>
#include <cstdio>
#include "triton/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"

const char* src =
R"(
const tunable int32 TM;
const tunable int32 TN;
const tunable int32 TK;

void matmul(restrict read_only fp32 *a, restrict read_only fp32 *b, fp32 *c,
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
    C = dot(a, b, C);
    pa = pa + TK*M;
    pb = pb + TK*K;
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


template<class T>
void simple_gemm(std::vector<T> &c, const std::vector<T> &a, const std::vector<T> &b, size_t M, size_t N, size_t K){
  for(size_t m = 0; m < M; m++)
  for(size_t n = 0; n < N; n++){
    T acc = 0;
    for(size_t k = 0; k < K; k++)
      acc += a[m + k*M] * b[n + k*N];
    c[m + n*M] = acc;
  }
}

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();

  // matrix multiplication parameters
  size_t M = 128, N = 128, K = 128;
  size_t bound = 8;
  std::vector<float> hc(M*N);
  std::vector<float> rc(M*N);
  std::vector<float> ha(M*K);
  std::vector<float> hb(K*N);
  srand(0);
  for(size_t i = 0; i < ha.size(); i++)
    ha[i] = 1;
  for(size_t i = 0; i < hb.size(); i++)
    hb[i] = 1;
  for(size_t i = 0; i < hc.size(); i++)
    hc[i] = 0;
  triton::driver::buffer dc(context, hc.size()*4);
  triton::driver::buffer da(context, ha.size()*4);
  triton::driver::buffer db(context, hb.size()*4);
  triton::driver::stream stream(context);
  stream.write(da, true, 0, ha);
  stream.write(db, true, 0, hb);
  stream.write(dc, true, 0, hc);
  stream.synchronize();

  // benchmark a given matrix multiplication kernel
  auto benchmark = [&](triton::driver::kernel kernel,
                       triton::jit::launch_information info) {
    kernel.setArg(0, da);
    kernel.setArg(1, db);
    kernel.setArg(2, dc);
    kernel.setArg(3, M);
    kernel.setArg(4, N);
    kernel.setArg(5, K);
    kernel.setArg(6, bound);
    unsigned TM = info.global_range_size[0];
    unsigned TN = info.global_range_size[1];
    unsigned nthreads = info.num_threads;
    stream.enqueue(kernel, {(M + TM - 1)/TM, (N + TN - 1)/TN, 1}, {nthreads, 1, 1});
    stream.synchronize();
    return float(0);
  };


//  std::vector<unsigned> params = {
//    // a0
//    2, 8, 1, 16,
//    // b0
//    4, 4, 1, 16,
//    // c
//    2, 4, 8, 4, 1, 1,
//    // a1
//    2, 4, 1, 8,
//    // b1
//    1, 8, 1
//  };

  // just-in-time compile source-code
  std::vector<unsigned> params = {
    // a0
    8, 2, 16,
    // b0
    4, 4, 16,
    // c
    8, 4, 2, 4,
    // a1
    4, 2, 8,
    // b1
    8, 1
  };
  triton::jit jit(context);
  jit.add_module(src, params);
  jit.autotune(src, benchmark);
  triton::driver::kernel kernel = jit.get_function("matmul");
  triton::jit::launch_information info = jit.get_launch_info("matmul");
  benchmark(kernel, info);
  stream.read(dc, true, 0, hc);
  simple_gemm(rc, ha, hb, M, N, K);
  for(size_t i = 0; i < M*N; i++)
    if(std::abs(hc[i] - rc[i])/std::max(hc[i], rc[i]) > 1e-4){
      std::cout << i << " " << hc[i] << " " << rc[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  std::cout << "Pass!" << std::endl;
}
