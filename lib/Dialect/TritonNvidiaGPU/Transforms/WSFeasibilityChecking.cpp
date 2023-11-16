/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace mlir {

namespace ttng = triton::nvidia_gpu;

namespace {

class TritonGPUWSFeasibilityCheckingPass
    : public TritonGPUWSFeasibilityCheckingBase<
          TritonGPUWSFeasibilityCheckingPass> {
public:
  TritonGPUWSFeasibilityCheckingPass() = default;
  TritonGPUWSFeasibilityCheckingPass(int computeCapability) {
    this->computeCapability = computeCapability;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int wsSupported = isWSSupported(mod, this->computeCapability);
    auto i32_ty = IntegerType::get(mod->getContext(), 32);
    mod->setAttr(ttng::TritonNvidiaGPUDialect::getWSSupportedAttrName(),
                 IntegerAttr::get(i32_ty, llvm::APInt(32, wsSupported)));
    if (wsSupported == 0) {
      mod->walk([](triton::FuncOp func) {
        llvm::errs() << "Warning: kernel \'" << func.getName()
                     << "\' cannot be warp specialized and will fall back to "
                        "the unspecialized version...\n";
      });
    }
    // TODO: remove it when unspecialzied kernel supports NUM_CTAS>1
    if (mod->getAttrOfType<IntegerAttr>("triton_gpu.num-ctas").getInt() > 1 &&
        wsSupported == 0)
      llvm::report_fatal_error(
          "Currently unspecialized kernel doesn't support NUM_CTAS > 1");
  }
};

} // namespace

std::unique_ptr<Pass>
createTritonNvidiaGPUWSFeasibilityCheckingPass(int computeCapability) {
  return std::make_unique<TritonGPUWSFeasibilityCheckingPass>(
      computeCapability);
}

} // namespace mlir
