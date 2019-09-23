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
  std::vector<config_t> configs;
  std::vector<int> x_idx = {0, 1, 2};
  do {
    std::vector<int> y_idx = {0, 1, 2};
    do {
      configs.push_back(config_t{{64, 64, 32}, {16, 4, 8}, x_idx, y_idx});
      configs.push_back(config_t{{64, 64, 32}, {8, 16, 2}, x_idx, y_idx});
      configs.push_back(config_t{{64, 64, 32}, {32, 2, 2}, x_idx, y_idx});
      configs.push_back(config_t{{64, 64, 32}, {16, 64, 4}, x_idx, y_idx});

    } while(std::next_permutation(y_idx.begin(), y_idx.end()));
  } while(std::next_permutation(x_idx.begin(), x_idx.end()));
  // testing
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


