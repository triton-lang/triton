#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_PASSES_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_PASSES_H_

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

} // namespace mlir

namespace mlir::triton {

#define GEN_PASS_DECL
#include "TritonAMDGPUToLLVM/Passes.h.inc"

} // namespace mlir::triton

namespace mlir::triton::AMD {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertWarpPipelinePass(StringRef gfxArch);
std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPUConvertWarpSpecializeToLLVMPass(StringRef gfxArch);

// Returns true if `module` has warp-pipeline stages for the warp-pipeline
// conversion pass to convert. Call it before that pass, which removes the
// stage markers.
bool hasWarpPipeline(ModuleOp module);

} // namespace mlir::triton::AMD

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonAMDGPUToLLVMPass(StringRef gfxArch, bool ftz, bool cuMode);
std::unique_ptr<OperationPass<ModuleOp>>
createConvertBuiltinFuncToLLVMPass(StringRef gfxArch, bool ftz);

std::unique_ptr<OperationPass<ModuleOp>>
createAllocateAMDGPUSharedMemoryPass(StringRef arch);

#define GEN_PASS_REGISTRATION
#include "TritonAMDGPUToLLVM/Passes.h.inc"

} // namespace mlir::triton

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_PASSES_H_
