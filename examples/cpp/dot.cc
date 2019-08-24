#include <cstring>
#include <sstream>
#include <cstdio>
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/tools/bench.hpp"
#include "triton/external/half.hpp"
#include "triton/runtime/function.h"
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

template<class T, bool AT, bool BT>
static void cpu_ref(std::vector<T> &c, const std::vector<T> &a, const std::vector<T> &b,
                    size_t M, size_t N, size_t K){
  for(size_t m = 0; m < M; m++)
  for(size_t n = 0; n < N; n++){
    float acc = 0;
    for(size_t k = 0; k < K; k++)
      acc = acc + (AT ? a[k + m*K] : a[m + k*M]) * (BT ? b[n + k*N] : b[k + n*K]);
    c[m + n*M] = static_cast<T>(acc);
  }
}

template<class T>
void cpu_ref(bool AT_, bool BT_, size_t M, size_t N, size_t K,
             std::vector<T> &c, const std::vector<T> &a, const std::vector<T> &b) {
  if(AT_ && BT_)
    cpu_ref<T, true, true>(c, a, b, M, N, K);
  else if(AT_ && !BT_)
    cpu_ref<T, true, false>(c, a, b, M, N, K);
  else if(!AT_ && BT_)
    cpu_ref<T, false, true>(c, a, b, M, N, K);
  else
    cpu_ref<T, false, false>(c, a, b, M, N, K);
}



std::string src(bool AT, bool BT, std::string a_ty, std::string b_ty, std::string c_ty, int align_lda, int align_ldb) {
  std::string ZS = "1";
  std::string AS0 = "TM", AS1 = "TK";
  std::string BS0 = "TK", BS1 = "TN";
  std::string XAS0 = "TM", XAS1 = "TK / " + ZS, XAS2 = ZS;
  std::string XBS0 = "TK / " + ZS, XBS1 = ZS, XBS2 = "TN";
  std::string bca0 = "[newaxis, :]", bca1 = "[:, newaxis]";
  std::string bcb0 = "[:, newaxis]", bcb1 = "[newaxis, :]";
  std::string lda0 = "*lda", lda1 = "";
  std::string ldb0 = "", ldb1 = "*ldb";
  std::string usea = AT ? "^a" : "a";
  std::string useb = BT ? "^b" : "b";
  if(AT){
    std::swap(AS0, AS1);
    std::swap(XAS0, XAS1);
    std::swap(XAS1, XAS2);
    std::swap(bca0, bca1);
    std::swap(lda0, lda1);
  }
  if(BT){
    std::swap(BS0, BS1);
    std::swap(XBS1, XBS2);
    std::swap(XBS0, XBS1);
    std::swap(bcb0, bcb1);
    std::swap(ldb0, ldb1);
  }
  std::string AS = AS0 + ", " + AS1;
  std::string BS = BS0 + ", " + BS1;
  std::string XCS = "TM, TN";
  std::string align_lda_str = "multipleof(" + std::to_string(align_lda) + ")";
  std::string align_ldb_str = "multipleof(" + std::to_string(align_ldb) + ")";
  std::string res =
R"(
#define bool _Bool
#define true 1
#define false 0
#define __bool_true_false_are_defined 1

#define __readonly      __attribute__((readonly))
#define __writeonly     __attribute__((writeonly))
#define __noalias       __attribute__((noalias))
#define __aligned(A)    __attribute__((aligned(A)))
#define __multipleof(A) __attribute__((multipleof(A)))

extern int get_program_id(int);

void matmul()" + a_ty + R"( * A __noalias __readonly __aligned(16),
            )" + b_ty + R"( * B __noalias __readonly __aligned(16),
            )" + c_ty + R"( * C __noalias __readonly __aligned(16),
            int M, int N, int K,
            int lda __multipleof(8),
            int ldb __multipleof(8),
            int ldc) {
  int ridx = get_program_id(0);
  int ridy = get_program_id(1);
  int rxa[TM] = ridx * TM + 0 ... TM;
  int ryb[TN] = ridy * TN + 0 ... TN;
  int rka[TK] = 0 ... TK;
  int rkb[TK] = 0 ... TK;
  float xc[)" + XCS + R"(] = 0;
  )" + a_ty + R"(* pa[)" + AS + "] = A + rka" + bca0 + lda0 + " + rxa" + bca1 + lda1 + R"(;
  )" + b_ty + R"(* pb[)" + BS + "] = B + rkb" + bcb0 + ldb0 + " + ryb" + bcb1 + ldb1 + R"(;
  )" + a_ty + R"( a[)" + AS + R"(] = *pa;
  )" + b_ty + R"( b[)" + BS + R"(] = *pb;
  for(int k = K; k > 0; k = k - TK){
    xc = )" + usea + " @ " + useb + R"( + xc;
    pa = pa + TK)" + lda0 + R"(;
    pb = pb + TK)" + ldb0 + R"(;
    a = *pa;
    b = *pb;
  }
  int rxc[TM] =  ridx * TM + (0 ... TM);
  int ryc[TN] =  ridy * TN + (0 ... TN);
  )" + c_ty + R"(* pc[TM, TN] = C + ryc[newaxis, :]*ldc + rxc[:, newaxis];
  )" + c_ty + R"( c[TM, TN] = xc;
  bool checkc0[TM] = rxc < M;
  bool checkc1[TN] = ryc < N;
  bool checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  *pc = c;
}
)";

  return res;
}

struct perf_t {
  double triton;
  double cublas;
};

namespace drv = triton::driver;
namespace rt = triton::runtime;

perf_t do_bench(drv::stream* stream, bool AT, bool BT, int32_t M, int32_t N, int32_t K){
  typedef half NumericT;
  std::string ty = "half";
  size_t dt_nbytes = sizeof(NumericT);
  drv::context* context = stream->context();
  std::vector<NumericT> hc(M*N);
  std::vector<NumericT> ha(M*K);
  std::vector<NumericT> hb(K*N);
  int32_t lda = AT ? K : M;
  int32_t ldb = BT ? N : K;
  int32_t ldc = M;
  srand(0);
  for(size_t i = 0; i < ha.size(); i++)
    ha[i] = static_cast<NumericT>((double)rand()/RAND_MAX);
  for(size_t i = 0; i < hb.size(); i++)
    hb[i] = static_cast<NumericT>((double)rand()/RAND_MAX);
  for(size_t i = 0; i < hc.size(); i++)
    hc[i] = static_cast<NumericT>((double)0);
  drv::buffer* dc = drv::buffer::create(context, hc.size()*dt_nbytes);
  drv::buffer* da = drv::buffer::create(context, ha.size()*dt_nbytes);
  drv::buffer* db = drv::buffer::create(context, hb.size()*dt_nbytes);
  stream->write(da, true, 0, ha);
  stream->write(db, true, 0, hb);
  stream->write(dc, true, 0, hc);
  stream->synchronize();
  // run
  rt::function::options_space_t opt;
  opt.defines.push_back({"TM", {"128"}});
  opt.defines.push_back({"TN", {"128"}});
  opt.defines.push_back({"TK", {"32"}});
  opt.num_warps = {1, 2, 4, 8};
  rt::function function(src(AT, BT, ty, ty, ty, 8, 8), opt);

  auto ceil = [](size_t x, size_t y) { return (x + y - 1) / y; };
  auto grid = [&](const rt::function::options_t& x) { return rt::grid_t{ceil(M, x.D<int>("TM")), ceil(N, x.D<int>("TN")), 1}; };

  auto tflops = [&](double nanosec) { return 2.*M*N*K / nanosec * 1e-3; };
  perf_t res;
  res.triton = tflops(triton::tools::bench([&]() { function({da, db, dc, M, N, K, lda, ldb, ldc}, grid, stream);}, stream));
  res.cublas = 0;

  // test
  stream->synchronize();
  stream->read(dc, true, 0, hc);
  std::vector<NumericT> rc(hc.size());
  cpu_ref(AT, BT, M, N, K, rc, ha, hb);
  for(size_t i = 0; i < M*N; i++)
    if(std::isinf(hc[i]) || std::isnan(hc[i]) || std::abs(hc[i] - rc[i])/std::max(hc[i], rc[i]) > 1e-2){
      std::cout << i << " " << hc[i] << " " << rc[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  std::cout << hc[0] << " " << std::endl;
  std::cout << "Pass!" << std::endl;

  // clean-up
  delete dc;
  delete da;
  delete db;
  return res;
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
    {false, true, 128, 128, 128}
//    {false, true, 128, 128, 128},
//    {false, false, 128, 128, 128},
//    {true, false, 128, 128, 128},
//    {true, true, 128, 128, 128}
//    {false, true, 32768, 256, 512}
//    {true,  false, 8192, 512, 512},
//    {true,  true,  8192, 512, 512}
  };
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context);
  // does the work
  for(config_t c: configs){
    perf_t perf = c.perf(stream);
    std::cout << "// " << c.repr() << ", " << perf.triton << ", " << perf.cublas << std::endl;
  }
}
