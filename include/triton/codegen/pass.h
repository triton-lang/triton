#ifndef _TRITON_CODEGEN_PASS_H_
#define _TRITON_CODEGEN_PASS_H_


#include <memory>

namespace llvm{
  class Module;
  class LLVMContext;
}

namespace triton{

namespace ir{
  class module;
}
namespace driver{
  class device;
  class module;
  class kernel;
}
}

namespace triton{
namespace codegen{

// TODO:
// There should be a proper pass manager there!
std::unique_ptr<llvm::Module> add_passes_to_emit_bin(ir::module &ir, llvm::LLVMContext& ctx,
                                                     int sm, int num_warps,
                                                     int num_stages, bool force_nc_cache);


}
}

#endif
