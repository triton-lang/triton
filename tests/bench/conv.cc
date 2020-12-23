#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "conv.h"

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context->backend());
  // shapes to benchmark
  typedef std::tuple<int, int, int, int, int, int, int, int, int, int, int> config_t;
  std::vector<config_t> configs = {
//      {1, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1},
//      {1, 56, 56, 128, 128, 3, 3, 1, 1, 1, 1},
//      {1, 56, 56, 256, 256, 3, 3, 1, 1, 1, 1},
//      {1, 56, 56, 384, 384, 3, 3, 1, 1, 1, 1},
//      {1, 56, 56, 512, 512, 3, 3, 1, 1, 1, 1},
//      {1, 56, 56, 768, 768, 3, 3, 1, 1, 1, 1},
//      {1, 56, 56, 1024, 1024, 3, 3, 1, 1, 1, 1},

//      {1, 8, 8, 256, 256, 3, 3, 1, 1, 1, 1},
//      {1, 16, 16, 256, 256, 3, 3, 1, 1, 1, 1},
//      {1, 32, 32, 256, 256, 3, 3, 1, 1, 1, 1},
//      {1, 64, 64, 256, 256, 3, 3, 1, 1, 1, 1},
      {1, 64, 64, 4096, 4096, 1, 1, 0, 0, 1, 1},
//      {1, 256, 256, 256, 256, 3, 3, 1, 1, 1, 1}




  };
  int Z, H, W, CO, CI, R, S, pad_h, pad_w, stride_h, stride_w;
  for(const auto& c: configs){
    std::tie(Z, H, W, CO, CI, R, S, pad_h, pad_w, stride_h, stride_w) = c;
    std::cout << "// " << c ;
    for(auto perf: bench_conv(context, stream, HALF, Z, H, W, CO, CI, R, S, pad_h, pad_w, stride_h, stride_w))
      std::cout << ", " << perf << std::flush;
    std::cout << std::endl;
  }
}
