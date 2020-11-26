#include <iostream>
#include <tuple>
#include "copy.h"
#include "triton/driver/backend.h"


int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context->backend());
  // shapes to benchmark
  typedef std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> config_t;
  std::vector<config_t> configs = {
    {{4096*4096}, {0}, {0}},
    {{4096, 4096}, {0, 1}, {0, 1}},
    {{4096, 4096}, {0, 1}, {1, 0}},
    {{4096, 4096}, {1, 0}, {0, 1}},
    {{4096, 4096}, {1, 0}, {1, 0}},
    {{256, 256, 256}, {0, 1, 2}, {0, 1, 2}},
    {{256, 256, 256}, {0, 1, 2}, {0, 2, 1}},
    {{256, 256, 256}, {1, 0, 2}, {1, 2, 0}},
    {{256, 256, 256}, {1, 2, 0}, {1, 0, 2}}
//    {{256, 256, 256}, {2, 0, 1}, {0, 1, 2}},
//    {{256, 256, 256}, {2, 1, 0}, {0, 2, 1}}
  };
  // does the work
  std::vector<int32_t> shape;
  std::vector<int32_t> ord_x, ord_y;
  for(const auto& c: configs){
    std::tie(shape, ord_x, ord_y) = c;
    std::cout << "// " << c << std::flush;
    for(auto perf: bench_copy_nd(context, stream, HALF, shape, ord_x, ord_y))
      std::cout << ", " << perf << std::flush;
    std::cout << std::endl;
  }
}
