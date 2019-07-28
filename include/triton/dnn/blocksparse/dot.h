#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include "triton/dnn/base.h"
#include <string>

namespace triton{
namespace dnn{
namespace blocksparse{


class dot: public base {
private:
  void enqueue_impl(driver::stream *stream, driver::kernel *kernel,
                    std::vector<driver::buffer*> args,
                    triton::runtime::launch_information info);
  // number of flops
  virtual size_t num_flops() const;
  // comparison for maps
  virtual bool operator<(const base& other) const;
  // default parameters
  virtual std::vector<params_t> search_space() const;
  virtual params_t heuristics() const;

public:
  // constructor
  dot(int32_t M, int32_t N, int32_t K);
  // triton-c source
  virtual void triton_c_src(std::ostream &os) const;
  // clone
  virtual base* clone() const;

private:
  std::string ab_ty_;
  std::string c_ty_;
  int32_t M_;
  int32_t N_;
  int32_t K_;
};

}
}
}
