#ifndef TDL_INCLUDE_JIT_H
#define TDL_INCLUDE_JIT_H

#include <string>
#include <memory>
#include "llvm/IR/LLVMContext.h"
#include "triton/ir/context.h"
#include "triton/driver/module.h"
#include "triton/driver/kernel.h"

namespace llvm {
  class Module;
}

namespace triton {

namespace ir {
class module;
class context;
}

class jit {
private:
  void init_llvm();
  std::string compute_data_layout(bool is64Bit = true, bool UseShortPointers = true);
  std::unique_ptr<llvm::Module> make_llvm_module(triton::ir::module &module, const std::vector<unsigned>& params);
  std::unique_ptr<ir::module> make_triton_module(const std::string &src);

public:
  jit(driver::context context);
  void add_module(ir::module &module, const std::vector<unsigned>& params = {});
  void add_module(const std::string &src, const std::vector<unsigned>& params = {});
  driver::kernel get_function(const std::string &name);

private:
  std::vector<driver::module> modules_;
  driver::context driver_context_;
  llvm::LLVMContext llvm_context_;
  ir::context triton_context_;
};


}

#endif
