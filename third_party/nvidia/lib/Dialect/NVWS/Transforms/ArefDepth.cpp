/*
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <memory>
#include <optional>

// #define GEN_PASS_CLASSES
// #include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace mlir {

#define GEN_PASS_DEF_NVWSAREFDEPTHPASS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-aref-depth"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
using namespace triton::nvws;

} // namespace

class NVWSArefDepthPass
    : public impl::NVWSArefDepthPassBase<NVWSArefDepthPass> {

  using impl::NVWSArefDepthPassBase<NVWSArefDepthPass>::NVWSArefDepthPassBase;

private:
  LogicalResult arefAssignDepth(tt::FuncOp funcOp, int accDepth,
                                int tmemLhsDepth) {

    auto getParentWgOp = [](ttng::ArefPutEnterOp op) -> ttng::WarpGroupOp {
      auto block = op->getBlock();
      while (block && block->getParentOp() &&
             !isa<triton::nvidia_gpu::WarpGroupOp>(block->getParentOp()))
        block = block->getParentOp()->getBlock();
      assert(block);
      assert(block->getParentOp());
      return cast<ttng::WarpGroupOp>(block->getParentOp());
    };

    SmallVector<ttng::ArefCreateOp> arefOps;
    funcOp.walk([&](ttng::ArefCreateOp arefOp) {
      if (!arefOp->hasAttr("aref_mma_sync"))
        arefOps.push_back(arefOp);
    });

    SmallVector<Operation *> allocsToErase;
    for (auto arefOp : arefOps) {
      DenseSet<ttng::WarpGroupOp> wgOps;
      bool usedInLoop = true;
      for (auto user : arefOp->getUsers()) {
        usedInLoop = usedInLoop && user->getParentOfType<scf::ForOp>();
        // verify that aref is used in the loop
        if (auto putEnterOp = dyn_cast<ttng::ArefPutEnterOp>(user))
          wgOps.insert(getParentWgOp(putEnterOp));
      }
      // if not used in loop, do not update aref depth
      if (!usedInLoop)
        continue;
      // if aref is used in multiple wgOps, do not update aref depth
      if (wgOps.size() == 1) {
        auto groups = getGroups(*wgOps.begin());
        assert(groups.size() == 1);
        auto group = *groups.begin();
        int depth = 1;
        if (group == ATTR_WS_TMALOAD) {
          depth = numStages;
        } else if (group == ATTR_WS_MMA) {
          depth = accDepth;
        } else if (group == ATTR_WS_SIMT) {
          depth = tmemLhsDepth;
        } else if (group == ATTR_WS_EPILOGUE) {
          llvm_unreachable("ArefPut in epilogue is not supported");
        }
        if (depth == 1)
          continue;

        SmallVector<Value> allocOps;
        SmallVector<Type> arefTypes;

        OpBuilder builder(arefOp);
        for (auto opnd : arefOp.getOperands()) {
          auto arefBufType = cast<ttg::MemDescType>(opnd.getType());
          arefBufType = getArefbufMemDescType(
              getDataMemDescType(arefBufType, true), depth);
          auto oldAlloc = opnd.getDefiningOp();
          auto loc = oldAlloc->getLoc();
          Operation *newAlloc = createAlloc(builder, loc, arefBufType, Value());
          newAlloc->setAttr("aref_buffer", builder.getUnitAttr());
          allocOps.push_back(newAlloc->getResult(0));
          arefTypes.push_back(arefBufType);
          allocsToErase.push_back(oldAlloc);
        }
        auto arefTy = triton::nvidia_gpu::ArefType::get(
            builder.getContext(),
            ttg::TypeArrayAttr::get(builder.getContext(), arefTypes));
        auto newAref = builder.create<triton::nvidia_gpu::ArefCreateOp>(
            arefOp.getLoc(), arefTy, allocOps);
        arefOp.getResult().replaceAllUsesWith(newAref.getResult());
        arefOp.erase();
      }
    }
    for (auto alloc : allocsToErase)
      alloc->erase();
    return success();
  }

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();

    SmallVector<tt::FuncOp> funcOps;
    m.walk([&](tt::FuncOp funcOp) { funcOps.push_back(funcOp); });

    auto canDoubleBufferTMEM = [](ttg::MemDescType tmemDesc,
                                  std::optional<ttng::MMAv5OpInterface> mmaOp,
                                  int usedBlocks) {
      auto blockM = tmemDesc.getShape()[0];
      auto blockN = tmemDesc.getShape()[1];
      constexpr int numTMEMColumns = 512;
      constexpr int numTMEMRows = 128;
      if (usedBlocks + (blockM * blockN * 2) > numTMEMRows * numTMEMColumns) {
        return false;
      }
      if (mmaOp && isa<ttng::TCGen5MMAScaledOp>(*mmaOp) && blockN == 256) {
        return false;
      }
      return true;
    };

    for (auto funcOp : funcOps) {
      int accDepth = 1;
      ttg::MemDescType accumMemdesc;
      funcOp.walk([&](ttng::MMAv5OpInterface mma) {
        accumMemdesc = mma.getAccumulator().getType();
        if (canDoubleBufferTMEM(accumMemdesc, mma, 0))
          accDepth = 2;
      });

      int accUsedBlocks = accumMemdesc
                              ? accumMemdesc.getShape()[0] *
                                    accumMemdesc.getShape()[1] * accDepth
                              : 0;

      int tmemLhsDepth = 1;
      funcOp.walk([&](ttng::MMAv5OpInterface mma) {
        auto lhsMemDesc = cast<ttg::MemDescType>(mma.getA().getType());
        if (isa<ttng::TensorMemorySpaceAttr>(lhsMemDesc.getMemorySpace())) {
          bool tmemStoreBySIMTWG = false;
          if (auto arefGet = mma.getA().getDefiningOp<ttng::ArefGetEnterOp>()) {
            for (auto user : arefGet.getAref().getUsers()) {
              if (isOpInGroup(user, ATTR_WS_SIMT) &&
                  isa<ttng::ArefPutEnterOp>(user)) {
                tmemStoreBySIMTWG = true;
              }
            }
          }

          if (tmemStoreBySIMTWG &&
              canDoubleBufferTMEM(lhsMemDesc, std::nullopt, accUsedBlocks)) {
            tmemLhsDepth = 2;
          }
        }
      });

      if (arefAssignDepth(funcOp, accDepth, tmemLhsDepth).failed())
        signalPassFailure();
    }
  }
};

} // namespace mlir
