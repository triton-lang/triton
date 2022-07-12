#ifndef _TRITON_CODEGEN_PASS_H_
#define _TRITON_CODEGEN_PASS_H_


#include <memory>
#include "extern_lib.h"

namespace llvm{
  class Module;
  class LLVMContext;
}

namespace triton{

namespace codegen {
  class target;
}

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
std::unique_ptr<llvm::Module> add_passes_to_emit_bin(
    ir::module &ir, llvm::LLVMContext &ctx, codegen::target *target,
    int num_warps, int num_stages, int &shared_static,
    const ExternLibMap &extern_libs);
}
}

#endif
