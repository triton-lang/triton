#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include <string>

namespace triton{
namespace dnn{

class gemm {
public:
  static void init(driver::stream* stream, driver::buffer* locks);
  static void set_arg(driver::kernel *kernel,
                      driver::buffer *a, driver::buffer *b, driver::buffer *c,
                      int32_t M, int32_t N, int32_t K,
                      driver::buffer *locks, int32_t grid_0, int32_t grid_1);
  static std::vector<unsigned> default_params(bool AT, bool BT);
  static std::string src(bool AT, bool BT);
};

}
}
