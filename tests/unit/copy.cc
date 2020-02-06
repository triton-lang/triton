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
  // 1D
//  configs.push_back({{65536}, {32}, {0}, {0}});
  configs.push_back({{65536}, {128}, {0}, {0}});
  configs.push_back({{65536}, {512}, {0}, {0}});
  configs.push_back({{65536}, {1024}, {0}, {0}});
  // 2D
  configs.push_back({{256, 256}, {16, 16}, {0, 1}, {0, 1}});
  configs.push_back({{256, 256}, {16, 64}, {0, 1}, {0, 1}});
  configs.push_back({{256, 256}, {64, 16}, {0, 1}, {0, 1}});
  configs.push_back({{256, 256}, {64, 64}, {0, 1}, {0, 1}});
  configs.push_back({{256, 256}, {16, 16}, {0, 1}, {1, 0}});
  configs.push_back({{256, 256}, {16, 64}, {0, 1}, {1, 0}});
  configs.push_back({{256, 256}, {64, 16}, {0, 1}, {1, 0}});
  configs.push_back({{256, 256}, {64, 64}, {0, 1}, {1, 0}});
  configs.push_back({{256, 256}, {16, 16}, {1, 0}, {0, 1}});
  configs.push_back({{256, 256}, {16, 64}, {1, 0}, {0, 1}});
  configs.push_back({{256, 256}, {64, 16}, {1, 0}, {0, 1}});
  configs.push_back({{256, 256}, {64, 64}, {1, 0}, {0, 1}});
  configs.push_back({{256, 256}, {64, 64}, {1, 0}, {1, 0}});
  configs.push_back({{256, 256}, {16, 64}, {1, 0}, {1, 0}});
  configs.push_back({{256, 256}, {64, 16}, {1, 0}, {1, 0}});
  configs.push_back({{256, 256}, {64, 64}, {1, 0}, {1, 0}});
  // 3D
  std::vector<std::vector<int>> xx_idx = {{0, 1, 2}, {2, 1, 0}, {1, 0, 2}};
  std::vector<std::vector<int>> yy_idx = {{0, 1, 2}, {2, 1, 0}, {1, 0, 2}};
  for(const auto& x_idx: xx_idx)
  for(const auto& y_idx: yy_idx){
      configs.push_back({{64, 64, 32}, {16, 4, 8}, x_idx, y_idx});
      configs.push_back({{64, 64, 32}, {8, 16, 2}, x_idx, y_idx});
      configs.push_back({{64, 64, 32}, {32, 2, 2}, x_idx, y_idx});
      configs.push_back({{64, 64, 32}, {16, 64, 4}, x_idx, y_idx});
  }
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


