#ifndef TDL_INCLUDE_JIT_H
#define TDL_INCLUDE_JIT_H

#include <string>
#include <memory>
#include "llvm/IR/LLVMContext.h"
#include "triton/ir/context.h"
#include "triton/driver/module.h"
#include "triton/driver/kernel.h"
#include <functional>

namespace llvm {
  class Module;
}

namespace triton {

namespace codegen{
class tune;
}

namespace ir {
class module;
class context;
class metaparameter;
}

class jit {
public:
  struct launch_information{
    std::vector<unsigned> global_range_size;
    unsigned num_threads;
  };
  typedef std::function<unsigned(driver::kernel, launch_information)> benchmark_t;

private:
  std::string compute_data_layout(bool is_64bit = true, bool use_short_pointers = true);
  std::unique_ptr<llvm::Module> make_llvm_module(triton::ir::module &module, codegen::tune &tune);
  std::unique_ptr<ir::module> make_triton_module(const std::string &src);

public:
  jit(driver::context context);
  void autotune(ir::module &module, benchmark_t benchmark);
  void autotune(const std::string &src, benchmark_t benchmark);
  void add_module(ir::module &module, const std::vector<unsigned>& params = {});
  void add_module(const std::string &src, const std::vector<unsigned>& params = {});
  driver::kernel get_function(const std::string &name);
  launch_information get_launch_info(const std::string &name);

private:
  std::vector<driver::module> modules_;
  driver::context driver_context_;
  llvm::LLVMContext llvm_context_;
  ir::context triton_context_;
  std::map<std::string, launch_information> launch_info_map_;
};


}

#endif
