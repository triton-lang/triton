#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "dot.h"

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context->backend());
  // shapes to benchmark
  typedef std::tuple<std::vector<int>, bool, bool, int, int, int> config_t;
  std::vector<config_t> configs;
  for(auto ord: std::vector<std::vector<int>>{{1, 0}})
      for(auto x: std::vector<std::array<bool, 2>>{{false, true}, {false, false}, {true, false}, {true, true}}){
    std::vector<config_t> tmp = {
//      config_t{ord, x[0], x[1], 128, 128, 128},
//      config_t{ord, x[0], x[1], 256, 256, 256},
//      config_t{ord, x[0], x[1], 384, 384, 384},
//      config_t{ord, x[0], x[1], 512, 512, 512},
//      config_t{ord, x[0], x[1], 768, 768, 768},
      // config_t{ord, x[0], x[1], 1024, 1024, 1024},
//      config_t{ord, x[0], x[1], 1280, 1280, 1280},
//      config_t{ord, x[0], x[1], 1536, 1536, 1536},
//      config_t{ord, x[0], x[1], 2048, 2048, 2048},
     config_t{ord, x[0], x[1], 4096, 4096, 4096},

//      config_t{ord, x[0], x[1], 256, 16, 256},
//      config_t{ord, x[0], x[1], 512, 16, 512},
//      config_t{ord, x[0], x[1], 768, 16, 768},
//      config_t{ord, x[0], x[1], 1024, 16, 1024},
//      config_t{ord, x[0], x[1], 1280, 16, 1280},
//      config_t{ord, x[0], x[1], 1536, 16, 1536},
//      config_t{ord, x[0], x[1], 2048, 16, 2048},
//      config_t{ord, x[0], x[1], 3072, 16, 3072},
//      config_t{ord, x[0], x[1], 4096, 16, 4096},
//      config_t{ord, x[0], x[1], 5120, 16, 5120},
//      config_t{ord, x[0], x[1], 6144, 16, 6144},
//      config_t{ord, x[0], x[1], 7168, 16, 7168},

//      config_t{ord, x[0], x[1], 64, 64, 4096},
//      config_t{ord, x[0], x[1], 64, 64, 8192},
//      config_t{ord, x[0], x[1], 64, 64, 16384},
//      config_t{ord, x[0], x[1], 64, 64, 32768},
//      config_t{ord, x[0], x[1], 64, 64, 65536},
//      config_t{ord, x[0], x[1], 64, 64, 131072}

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
    for(auto perf: bench_dot(context, stream, HALF, AT, BT, M, N, K, ord, ord))
      std::cout << ", " << perf << std::flush;
    std::cout << std::endl;
  }
}
