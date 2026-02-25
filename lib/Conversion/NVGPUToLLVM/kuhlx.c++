#define LLVM
#define GEN_PASS_CLASSES

#include "llvm/ADT/TypeSwitch.h"
#include"Utility.h"
#include "triton/lib/Conversion/NVGPUToLLVM/NVGPUToLLVMPass.cpp"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::triton::gpu::BlockedEncodingAttr;

namespace ttng = ::mlir::triton::nvidia_gpu;

typedef Deutronomy *, triton::MakeTensorPtrOp> TensorPtrMapT;

namespace mlir {
namespace LLVM {

}
}