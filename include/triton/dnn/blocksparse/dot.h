#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include "triton/dnn/base.h"
#include <string>

namespace triton{
namespace dnn{
namespace blocksparse{

enum op_t{
  FPROP,
  BPROP,
  WGRAD
};

class dot: public base {
private:
  void enqueue_impl(driver::stream *stream, driver::kernel *kernel,
                    std::vector<driver::buffer*> args,
                    triton::runtime::launch_information info);
  // number of flops
  size_t num_flops() const;
  // retuning parameters
  std::vector<int64_t> retune_params() const;
  // default parameters
  std::vector<params_t> search_space() const;
  params_t heuristics() const;
  // init
  void init_impl(driver::stream *stream, driver::cu_module *module, triton::runtime::launch_information info);
  // deinit
  void deinit_impl();
public:
  // constructor
  dot(int32_t N, int32_t K, int32_t S, int32_t C, const std::string &ty, int32_t BS, int32_t nlocks, int32_t nblocks, op_t op = FPROP);
  // triton-c source
  void triton_c_src(std::ostream &os) const;
  // locks
  driver::buffer* get_locks() const;
  // clone
  base* clone() const;

private:
  std::string ab_ty_;
  std::string c_ty_;
  int32_t N_;
  int32_t S_;
  int32_t C_;
  int32_t K_;
  int32_t BS_;
  int32_t nlocks_;
  int32_t nblocks_;
  std::shared_ptr<driver::buffer> locks_;
  op_t op_;
};

}
}
}
