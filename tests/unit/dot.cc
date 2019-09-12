#include <iomanip>
#include <cstring>
#include <sstream>
#include <cstdio>
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/tools/bench.hpp"
#include "triton/external/half.hpp"
#include "triton/runtime/function.h"
#include "src/dot.h"
#include "cuda/cublas.h"
#include "util.h"

namespace drv = triton::driver;
namespace rt = triton::runtime;

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


bool do_test(drv::stream* stream, bool AT, bool BT, int32_t M, int32_t N, int32_t K, int32_t TM, int32_t TN, int32_t TK, int nwarp){
  typedef float NumericT;
  std::string ty = "float";
  size_t dt_nbytes = sizeof(NumericT);
  drv::context* context = stream->context();
  std::vector<NumericT> hc(M*N);
  std::vector<NumericT> ha(M*K);
  std::vector<NumericT> hb(K*N);
  int32_t lda = AT ? K : M;
  int32_t ldb = BT ? N : K;
  int32_t ldc = M;
  srand(0);
  init_rand(ha);
  init_rand(hb);
  init_rand(hc);
  auto dc = std::shared_ptr<drv::buffer>(drv::buffer::create(context, hc.size()*dt_nbytes));
  auto da = std::shared_ptr<drv::buffer>(drv::buffer::create(context, ha.size()*dt_nbytes));
  auto db = std::shared_ptr<drv::buffer>(drv::buffer::create(context, hb.size()*dt_nbytes));
  stream->write(&*da, true, 0, ha);
  stream->write(&*db, true, 0, hb);
  stream->write(&*dc, true, 0, hc);
  stream->synchronize();
  // run
  rt::function::options_space_t opt;
  opt.defines.push_back({"TYPE", {ty}});
  opt.defines.push_back({"AT", {AT?"1":"0"}});
  opt.defines.push_back({"BT", {BT?"1":"0"}});
  opt.defines.push_back({"TM", {std::to_string(TM)}});
  opt.defines.push_back({"TN", {std::to_string(TN)}});
  opt.defines.push_back({"TK", {std::to_string(TK)}});
  opt.num_warps = {nwarp};
  rt::function function(src::dot, opt);
  try {
    function({&*da, &*db, &*dc, M, N, K, lda, ldb, ldc}, grid2d(M, N), stream);
  } catch (const std::runtime_error& e) {
    return true;
  }
  // test
  stream->read(&*dc, true, 0, hc);
  std::vector<NumericT> rc(hc.size());
  cpu_ref(AT, BT, M, N, K, rc, ha, hb);
  return diff(hc, rc);
}

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context);
  // shapes to benchmark
  typedef std::tuple<bool, bool, int, int, int, int, int, int, int> config_t;
  std::vector<config_t> configs;
  for(bool AT: std::array<bool, 2>{false, true})
  for(bool BT: std::array<bool, 2>{false, true})
  for(int TM: std::vector<int>{16, 128})
  for(int TN: std::vector<int>{16, 128})
  for(int TK: std::vector<int>{16, 32})
  for(int nwarps: std::vector<int>{1, 2, 4, 8}){
    configs.push_back(config_t{AT, BT, 128, 128, 128, TM, TN, TK, nwarps});
  }
  // does the work
  bool AT, BT;
  int M, N, K, TM, TN, TK, nwarp;
  for(const auto& c: configs){
    std::tie(AT, BT, M, N, K, TM, TN, TK, nwarp) = c;
    std::cout << "Testing " << c << " ... " << std::flush;
    if(do_test(stream, AT, BT, M, N, K, TM, TN, TK, (size_t)nwarp))
      std::cout << " Pass! " << std::endl;
    else
      std::cout << " Fail! " << std::endl;
  }
}
