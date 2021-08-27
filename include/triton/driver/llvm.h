#include <string>
#include "triton/driver/dispatch.h"

namespace llvm{
class Module;
}

namespace triton{
namespace driver{

void init_llvm();
std::string llir_to_ptx(llvm::Module* module, int cc, int version);
CUmodule ptx_to_cumodule(const std::string& ptx, int cc);

}
}
