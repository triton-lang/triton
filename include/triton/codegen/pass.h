#ifndef _TRITON_CODEGEN_PASS_H_
#define _TRITON_CODEGEN_PASS_H_


#include <memory>

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
void add_passes_to_emit_bin(ir::module &ir, driver::device* dev, int num_warps, int num_stages, bool force_nc_cache,
                            driver::module*& mod, driver::kernel*& ker, size_t& shared_mem);


}
}

#endif
