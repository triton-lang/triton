#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "dot.h"

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context);
  // shapes to benchmark
  typedef std::tuple<bool, bool, int, int, int> config_t;
  std::vector<config_t> configs;
  for(auto x: std::vector<std::array<bool, 2>>{{false, false}, {false, true},
                                               {true, false}, {true, true}}){
    std::vector<config_t> tmp = {
      config_t{x[0], x[1], 2048, 2048, 2048},
//      config_t{x[0], x[1], 16, 2048, 2048},
//      config_t{x[0], x[1], 32, 2048, 2048},
//      config_t{x[0], x[1], 64, 2048, 2048},
//      config_t{x[0], x[1], 128, 2048, 2048},
//      config_t{x[0], x[1], 7000, 2048, 2048},
//      config_t{x[0], x[1], 16, 4096, 4096},
//      config_t{x[0], x[1], 32, 4096, 4096},
//      config_t{x[0], x[1], 64, 4096, 4096},
//      config_t{x[0], x[1], 128, 4096, 4096},
//      config_t{x[0], x[1], 7000, 4096, 4096}
    };
    configs.insert(configs.end(), tmp.begin(), tmp.end());
  }
  // does the work
  bool AT, BT;
  int32_t M, N, K;
  for(const auto& c: configs){
    std::tie(AT, BT, M, N, K) = c;
    std::cout << "// " << AT << " " << BT << " " << M << " " << N << " " << K << std::flush;
    for(auto perf: bench_dot(stream, FLOAT, AT, BT, M, N, K))
      std::cout << ", " << perf << std::flush;
    std::cout << std::endl;
  }
}
