#include "triton/driver/dispatch.h"
#include <string>

namespace llvm {
class Module;
}

namespace triton {
namespace driver {

void init_llvm();
std::string path_to_ptxas(int &version);
std::string llir_to_ptx(llvm::Module *module, int cc, int version);
std::string ptx_to_cubin(const std::string &ptx, const std::string &ptxas_path,
                         int cc);
CUmodule ptx_to_cumodule(const std::string &ptx, int cc);
std::string llir_to_amdgpu(llvm::Module *module, const std::string &proc);
hipModule_t amdgpu_to_hipmodule(const std::string &path);

} // namespace driver
} // namespace triton
