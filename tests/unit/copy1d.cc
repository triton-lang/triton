#include <iostream>
#include <tuple>
#include "copy.h"
#include "triton/driver/backend.h"


int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context);
  // shapes to benchmark
  typedef std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>> config_t;
  std::vector<config_t> configs = {
//    {{65536}, {32}, {0}, {0}},
    {{65536}, {128}, {0}, {0}},
    {{65536}, {512}, {0}, {0}},
    {{65536}, {1024}, {0}, {0}},
  };
  // does the work
  std::vector<int32_t> shape, tile;
  std::vector<int32_t> ord_x, ord_y;
  bool result = true;
  for(const auto& c: configs){
    std::tie(shape, tile, ord_x, ord_y) = c;
    bool pass = test_copy_nd(stream, shape, tile, ord_x, ord_y);
    result = result && pass;
    std::cout << "// " << c << ", " << pass << std::endl;
  }
  return result;
}
