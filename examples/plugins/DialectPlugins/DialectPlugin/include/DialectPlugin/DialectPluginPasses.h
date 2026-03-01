#ifndef DIALECTPLUGIN_DIALECTPLUGINPASSES_H
#define DIALECTPLUGIN_DIALECTPLUGINPASSES_H

#include "DialectPlugin/DialectPluginDialect.h"
#include "DialectPlugin/DialectPluginOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {

class ModuleOp;

namespace triton {
namespace plugin {
class PluginTypeConverter : public TypeConverter {
public:
  PluginTypeConverter(MLIRContext *context, int numWarps, int threadsPerWarp,
                      int numCTAs);
  int getNumWarps() const { return numWarps; }
  int getThreadsPerWarp() const { return threadsPerWarp; }
  int getNumCTAs() const { return numCTAs; }

private:
  MLIRContext *context;
  int numWarps;
  int threadsPerWarp;
  int numCTAs;
};

#define GEN_PASS_DECL
#include "DialectPlugin/DialectPluginPasses.h.inc"

std::unique_ptr<OperationPass<ModuleOp>>
createConvertPluginGPUToLLVMPass(int32_t computeCapability = 80,
                                 int32_t ptxVersion = 80);
std::unique_ptr<OperationPass<ModuleOp>> createConvertPluginGPUToTritonGPUPass(
    int32_t num_warps = 4, int32_t threadsPerWarp = 32, int32_t numCTAs = 1);

#define GEN_PASS_REGISTRATION
#include "DialectPlugin/DialectPluginPasses.h.inc"
} // namespace plugin
} // namespace triton
} // namespace mlir

#endif
