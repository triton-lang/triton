#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include <iomanip>
#include <cstring>
#include <sstream>
#include <cstdio>
#include <tuple>
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/tools/bench.hpp"
#include "triton/external/half.hpp"
#include "triton/runtime/function.h"
#include <iomanip>
#include <cmath>
#include "triton/runtime/function.h"

namespace drv = triton::driver;
namespace rt = triton::runtime;

namespace src {

    const char *dot =
R"(
#define STM 8
#define STN 8

__global__ void dot(TYPE * A __noalias __readonly __aligned(16),
                    TYPE * B __noalias __readonly __aligned(16),
                    TYPE * C __noalias __aligned(16),
                    float alpha,
                    int M __retune,
                    int N __retune,
                    int K __retune __multipleof(16),
                    int lda __multipleof(8),
                    int ldb __multipleof(8),
                    int ldc __multipleof(8),
                    int* locks) {
      // prologue
      int pid = get_program_id(0);
      int pidz = get_program_id(2);
      int gridm = (M + TM - 1) / TM;
      int gridn = (N + TN - 1) / TN;
      int width = STM*gridn;
      int stm = pid / width;
      int RSTM  = min(gridm - stm*STM, STM);
      int stn =  (pid % width) / (RSTM*STN);
      int RSTN = min(gridn - stn*STN, STN);
      int laneid = pid % (RSTM * RSTN);
      int lanem = laneid / RSTN;
      int lanen = laneid % RSTN;
      int pidm = stm*STM + lanem;
      int pidn = stn*STN + lanen;
      int rm[TM] = pidm * TM + 0 ... TM;
      int rn[TN] = pidn * TN + 0 ... TN;

      // reduction splitting
      K           = K / TZ;
      int rk[TK]  = pidz * K + 0 ... TK;
      // pointers to operands
      int offa[TM, TK] = rk[newaxis, :] * STRIDE_AK + rm[:, newaxis] * STRIDE_AM;
      int offb[TK, TN] = rk[:, newaxis] * STRIDE_BK + rn[newaxis, :] * STRIDE_BN;
      TYPE* pa[TM, TK] = A + offa;
      TYPE* pb[TK, TN] = B + offb;
      // prefetches operands
      bool checka[TM, TK] = rk[newaxis, :] < K;
      bool checkb[TK, TN] = rk[:, newaxis] < K;
      TYPE a[TM, TK] = checka ? *pa : 0;
      TYPE b[TK, TN] = checkb ? *pb : 0;
      // reduction loop
      float acc[TM, TN] = 0;
      for(int k = K; k > 0; k -= TK){
        bool checka[TM, TK] = k > TK;
        bool checkb[TK, TN] = k > TK;
        pa += TK * STRIDE_AK;
        pb += TK * STRIDE_BK;
        TYPE anext[TM, TK] = *?(checka)pa;
        TYPE bnext[TK, TN] = *?(checkb)pb;
        acc += a @ b;
        a = anext;
        b = bnext;
//        __debug_barrier();
      }
      acc = acc * alpha;
      TYPE c[TM, TN] = acc;

      // epilogue
      int rcm[TM] = pidm * TM + 0 ... TM;
      int rcn[TN] = pidn * TN + 0 ... TN;
      int offc[TM, TN] = rcm[:, newaxis] * ldc + rcn[newaxis, :];
      TYPE* pc[TM, TN] = C + offc;
      bool checkc[TM, TN] = rcm[:, newaxis] < M &&
                            rcn[newaxis, :] < N;
#if (TZ==1)
      *?(checkc) pc = c;
#else
      // accumulate partial result using spin-locks
      int *plock  = locks + rid;
      int *pcount = plock + get_num_programs(0) * get_num_programs(1);
      for(int repeat = 1; repeat == 1; repeat = atomic_cas(plock, 0, 1));
      int count = *pcount;
      if(count == 0)
        *?(checkc) pc = c;
      else
        *?(checkc) pc = c + *?(checkc)pc;
      atomic_xchg(pcount, (count + 1) % TZ);
      atomic_xchg(plock, 0);
#endif
}
)";

}

enum dtype_t {
  FLOAT,
  HALF,
  DOUBLE
};

template<class T>
struct to_string;

template<> struct to_string<half_float::half>{
  static constexpr const char* value = "half";
};

template<> struct to_string<float>{
  static constexpr const char* value = "float";
};

template<> struct to_string<double>{
  static constexpr const char* value = "double";
};

template<class T>
float triton_dot(drv::context* context,  drv::stream* stream,
                 bool AT, bool BT,
                int32_t M, int32_t N, int32_t K){
  std::string ty = to_string<T>::value;
  size_t dt_nbytes = sizeof(T);
  drv::device* device = context->device();
  int32_t lda = AT ? K : M;
  int32_t ldb = BT ? N : K;
  int32_t ldc = N;
  std::vector<std::string> sa = { "1", "lda" };
  std::vector<std::string> sb = { "1", "ldb" };
  // inputs
  auto dc     = std::shared_ptr<drv::buffer>(drv::buffer::create(context, M*N*dt_nbytes));
  auto da     = std::shared_ptr<drv::buffer>(drv::buffer::create(context, M*K*dt_nbytes));
  auto db     = std::shared_ptr<drv::buffer>(drv::buffer::create(context, K*N*dt_nbytes));
  auto dlocks = std::shared_ptr<drv::buffer>(drv::buffer::create(context, 1024*1024*2*4));
  // initialize buffers
  std::vector<T> hc(M*N);
  std::vector<T> ha(M*K);
  std::vector<T> hb(K*N);
  for(size_t i = 0; i < ha.size(); i++)
    ha[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < hb.size(); i++)
    hb[i] = (float)rand()/RAND_MAX;
  stream->write(&*da, true, 0, ha);
  stream->write(&*db, true, 0, hb);
  // macros
  rt::options_t opt;
  opt.defines["STRIDE_AK"] = AT? "1"   : "lda";
  opt.defines["STRIDE_AM"] = AT? "lda" : "1";
  opt.defines["STRIDE_BK"] = BT? "ldb" : "1";
  opt.defines["STRIDE_BN"] = BT? "1"   : "ldb";
  opt.defines["TYPE"] = ty;
  opt.defines["TM"] = "128";
  opt.defines["TN"] = "128";
  opt.defines["TK"] = "64" ;
  opt.defines["TZ"] = "1";
  opt.num_warps = 4;
  // arguments
  std::stringstream oss;
  rt::add_arg(oss, *da->cu());
  rt::add_arg(oss, *db->cu());
  rt::add_arg(oss, *dc->cu());
  rt::add_arg(oss, (float)1);
  rt::add_arg(oss, M);
  rt::add_arg(oss, N);
  rt::add_arg(oss, K);
  rt::add_arg(oss, lda);
  rt::add_arg(oss, ldb);
  rt::add_arg(oss, ldc);
  rt::add_arg(oss, *dlocks->cu());
  // function
  rt::function function(src::dot, opt, device);
//  std::cout << function.get_kernels()[0].second->get_asm(rt::ASM_NV_PTX) << std::endl;
  // grid
  auto ceil = [](size_t x, size_t y) { return (x + y - 1) / y; };
  auto grid = [ceil, M, N](const rt::options_t& x) {
    return rt::kernel::grid_t{ceil(M, x.D<int>("TM"))*
                              ceil(N, x.D<int>("TN")),
                              (size_t)x.D<int>("TZ")};
  };

  // metrics
  auto tflops = [&](double nanosec) { return 2.*M*N*K / nanosec * 1e-3; };
  double triton_ns = triton::tools::bench([&]() { function(oss.str(), grid, stream);}, stream);
  return tflops(triton_ns);
}

float bench_dot(drv::context* context,  drv::stream* stream,
                bool AT, bool BT,
                int32_t M, int32_t N, int32_t K,
                dtype_t dtype) {
  switch(dtype){
    case HALF: return triton_dot<half_float::half>(context, stream, AT, BT, M, N, K);
    case FLOAT: return triton_dot<float>(context, stream, AT, BT, M, N, K);
    case DOUBLE: return triton_dot<double>(context, stream, AT, BT, M, N, K);
    default: return 0;
  }
}

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context->backend());
  // shapes to benchmark
  typedef std::tuple<bool, bool, int, int, int> config_t;
  std::vector<config_t> configs = {
    {false, false, 8192, 8192, 8192}
  };
  // does the work
  bool AT, BT;
  int32_t M, N, K;
  dtype_t dtype = HALF;
  for(const auto& c: configs){
    std::tie(AT, BT, M, N, K) = c;
    float tflops = bench_dot(context, stream, AT, BT, M, N, K, dtype);
    std::cout << "// " << AT << ", " << BT << ", " << M << ", " << N << ", " << K << ", " << tflops << std::endl;
  }
}
