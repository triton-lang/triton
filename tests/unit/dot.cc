#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "dot.h"
#include "util.h"

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context->backend());
  // shapes to test
  typedef std::tuple<dtype_t, bool, bool, int, int, int, int> config_t;
  std::vector<config_t> configs;
  for(dtype_t dtype: std::vector<dtype_t>{FLOAT, HALF})
  for(bool AT: std::vector<bool>{false, true})
  for(bool BT: std::vector<bool>{false, true}){
    // 1 warp
    configs.push_back({dtype, AT, BT, 16, 16, 16, 1});
    configs.push_back({dtype, AT, BT, 32, 16, 16, 1});
    configs.push_back({dtype, AT, BT, 16, 32, 16, 1});
    configs.push_back({dtype, AT, BT, 16, 16, 32, 1});
    configs.push_back({dtype, AT, BT, 32, 16, 32, 1});
    configs.push_back({dtype, AT, BT, 16, 32, 32, 1});
    if(dtype == HALF){
      configs.push_back({dtype, AT, BT, 16, 64, 16, 1});
      configs.push_back({dtype, AT, BT, 16, 16, 64, 1});
      configs.push_back({dtype, AT, BT, 64, 16, 64, 1});
      configs.push_back({dtype, AT, BT, 16, 64, 64, 1});
    }
    // 2 warps
    configs.push_back({dtype, AT, BT, 64, 32, 64, 2});
    configs.push_back({dtype, AT, BT, 32, 64, 64, 2});
    configs.push_back({dtype, AT, BT, 64, 32, 16, 2});
    configs.push_back({dtype, AT, BT, 32, 64, 16, 2});
    configs.push_back({dtype, AT, BT, 128, 32, 32, 2});
    configs.push_back({dtype, AT, BT, 32, 128, 32, 2});
    // 4 warps
    configs.push_back({dtype, AT, BT, 128, 64, 16, 4});
    configs.push_back({dtype, AT, BT, 64, 128, 16, 4});
    configs.push_back({dtype, AT, BT, 128, 32, 32, 4});
    configs.push_back({dtype, AT, BT, 32, 128, 32, 4});
    if(dtype == HALF){
      configs.push_back({dtype, AT, BT, 128, 32, 64, 4});
      configs.push_back({dtype, AT, BT, 32, 128, 64, 4});
    }
    // 8 warps
    configs.push_back({dtype, AT, BT, 128, 256, 16, 8});
    configs.push_back({dtype, AT, BT, 256, 128, 16, 8});
    if(dtype == HALF){
      configs.push_back({dtype, AT, BT, 256, 128, 32, 8});
      configs.push_back({dtype, AT, BT, 256, 128, 32, 8});
    }

  };
  // test
  dtype_t dtype;
  bool AT, BT;
  int M, N, K, TM, TN, TK, nwarp;
  for(const auto& c: configs){
    std::tie(dtype, AT, BT, TM, TN, TK, nwarp) = c;
    M = TM;
    N = TN;
    K = TK;
    std::cout << "Testing " << c << " ... " << std::flush;
    if(test_dot(context, stream, dtype, AT, BT, M, N, K, {0, 1}, {0, 1}, TM, TN, TK, (size_t)nwarp))
      std::cout << " Pass! " << std::endl;
    else{
      std::cout << " Fail! " << std::endl;
    }
  }
}
