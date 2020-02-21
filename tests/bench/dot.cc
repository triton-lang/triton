#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "dot.h"

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context);
  // shapes to benchmark
  typedef std::tuple<std::vector<int>, bool, bool, int, int, int> config_t;
  std::vector<config_t> configs;
  for(auto ord: std::vector<std::vector<int>>{{1, 0}})
    for(auto x: std::vector<std::array<bool, 2>>{{false, false}}){
    std::vector<config_t> tmp = {
//      config_t{ord, x[0], x[1], 512, 512, 512},
      config_t{ord, x[0], x[1], 2048, 2048, 2048},
//      config_t{ord, x[0], x[1], 127008, 768, 576},
//      config_t{ord, x[0], x[1], 8192, 8192, 8192}
//      config_t{ord, x[0], x[1], 16, 2048, 2048},
//      config_t{ord, x[0], x[1], 32, 2048, 2048},
//      config_t{ord, x[0], x[1], 64, 2048, 2048},
//      config_t{ord, x[0], x[1], 128, 2048, 2048},
//      config_t{ord, x[0], x[1], 7000, 2048, 2048},
//      config_t{ord, x[0], x[1], 16, 4096, 4096},
//      config_t{ord, x[0], x[1], 32, 4096, 4096},
//      config_t{ord, x[0], x[1], 64, 4096, 4096},
//      config_t{ord, x[0], x[1], 128, 4096, 4096},
//      config_t{ord, x[0], x[1], 7000, 4096, 4096}
    };
    configs.insert(configs.end(), tmp.begin(), tmp.end());
  }
  // does the work
  std::vector<int> ord;
  bool AT, BT;
  int32_t M, N, K;
  for(const auto& c: configs){
    std::tie(ord, AT, BT, M, N, K) = c;
    std::cout << "// " << c ;
    for(auto perf: bench_dot(stream, FLOAT, AT, BT, M, N, K, ord, ord))
      std::cout << ", " << perf << std::flush;
    std::cout << std::endl;
  }
}
