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

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/WSUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

// #include <fstream>
// #include <iostream>
// #include <memory>
// #include <stack>

#define GEN_PASS_CLASSES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-assign-groups"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;
using namespace triton::nvws;
namespace ttng = triton::nvidia_gpu;

class InitialAssignment {
public:
  void runOnOperation(ModuleOp op) {
    calculateNumWarps(op);
    visitModule(op);
  }

private:
  struct NumWarpsSpec {
    int load;
    int MMA;
    int epilogue;
  };
  NumWarpsSpec numWarps;

  void calculateNumWarps(ModuleOp m) {
    auto numWarpsAttr =
        mlir::cast<mlir::IntegerAttr>(m->getAttr(AttrNumWarpsName));
    const int numLoadWarps = 4;
    const int numMMAWarps = numWarpsAttr.getInt();
    const int numEpiWarps =
        shouldCreateEpilogueGroup(m) ? numWarpsAttr.getInt() : 0;
    numWarps = {numLoadWarps, numMMAWarps, numEpiWarps};
  }

  bool shouldCreateEpilogueGroup(ModuleOp m) {
    SmallVector<triton::nvidia_gpu::TMEMLoadOp> tmemLoadOps;

    m.walk(
        [&](triton::nvidia_gpu::TMEMLoadOp op) { tmemLoadOps.push_back(op); });

    // create epilogue group if there exist a tmem_load op which is outside the
    // loop enclosing the corresponding mmav5 op
    for (auto tmemLoadOp : tmemLoadOps) {
      auto mmaOp = getMMAv5Op(tmemLoadOp);
      if (mmaOp) {
        auto mmaLoop = mmaOp->getParentOfType<scf::ForOp>();
        if (mmaLoop && mmaLoop->getBlock() == tmemLoadOp->getBlock()) {
          // now check from this tmem_load op to the end of the enclosing block
          // to see if there is any op uses the forOp result
          for (Operation &op : llvm::make_early_inc_range(llvm::make_range(
                   std::next(tmemLoadOp->getIterator()),
                   mmaLoop->getBlock()->getTerminator()->getIterator()))) {
            for (auto result : mmaLoop->getResults()) {
              if (llvm::any_of(result.getUsers(),
                               [&](Operation *user) { return user == &op; })) {
                return false;
              }
            }
          }
          return true;
        }
      }
    }
    return false;
  }

  triton::nvidia_gpu::MMAv5OpInterface
  getMMAv5Op(triton::nvidia_gpu::TMEMLoadOp tmemLoadOp) {
    auto tmemAddr = tmemLoadOp->getOperand(0);
    for (auto user : tmemAddr.getUsers()) {
      if (auto mmaOp = dyn_cast<triton::nvidia_gpu::MMAv5OpInterface>(user)) {
        return mmaOp;
      }
    }
    return nullptr;
  };

  void visit(Operation *op) {
    if (auto moduleOp = llvm::dyn_cast<ModuleOp>(op)) {
      visitModule(moduleOp);
    } else if (auto funcOp = llvm::dyn_cast<FuncOp>(op)) {
      visitFunc(funcOp);
    } else if (auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
      visitFor(forOp);
    } else if (auto ifOp = llvm::dyn_cast<scf::IfOp>(op)) {
      visitIf(ifOp);
    } else {
      assignGroups(op);
    }
  }

  void visitModule(ModuleOp op) {
    for (auto func : op.getOps<FuncOp>()) {
      visit(func);
    }
  }

  void visitFunc(FuncOp op) {
    Block &body = op.getBody().front();
    for (auto it = body.begin(); it != body.end(); ++it) {
      visit(&*it);
    }
  }

  void visitFor(scf::ForOp forOp) {
    for (auto &op : forOp.getBody()->without_terminator()) {
      visit(&op);
    }
  }

  void visitIf(scf::IfOp ifOp) {
    if (auto thenBlock = ifOp.thenBlock()) {
      for (auto &op : *thenBlock) {
        visit(&op);
      }
    }
    if (auto elseBlock = ifOp.elseBlock()) {
      for (auto &op : *elseBlock) {
        visit(&op);
      }
    }
  }

  bool isMMAOperandLoadOp(Operation *op) {
    // Check if the given LoadOp can be lowered to cpasync
    if (!isa<LoadOp>(op))
      return false;

    LoadOp loadOp = cast<LoadOp>(op);
    auto resType = loadOp.getResult().getType();
    if (!isa<RankedTensorType>(resType)) {
      return false;
    }

    LocalAllocOp localAlloc;

    std::function<bool(Operation *)> isUsedByDot = [&](Operation *op) {
      // transitively check if one of the user of the op is a dot op
      if (isa<MMAv5OpInterface, WarpGroupDotOp>(op)) {
        return true;
      } else {
        if (isa<LocalAllocOp>(op)) {
          localAlloc = cast<LocalAllocOp>(op);
        }
        for (auto user : op->getUsers())
          if (isUsedByDot(user))
            return true;
        return false;
      }
    };

    // For now, allow cpasync only for loading dot operands
    // When we add support for a specialized warp group for loading additional
    // tensors used in the epilogue, we should relax this condition.
    if (!isUsedByDot(op))
      return false;

    auto sharedEnc =
        mlir::cast<SharedEncodingTrait>(localAlloc.getType().getEncoding());
    auto copyVecBytes =
        getCopyVecBytes(localAlloc.getSrc().getType(), sharedEnc);

    // At least 4 bytes needed for cpasync
    return copyVecBytes >= 4;
  }

  void assignGroups(Operation *op) {
    OpBuilder builder(op);

    // load ops placed in load warp group
    // FIXME: TMA can also be used for loading bias etc in the epilouge.
    // Those loads must be in a different group than the load group that
    // feeds MMA.
    if (isa<triton::DescriptorLoadOp, triton::DescriptorGatherOp>(op) ||
        isMMAOperandLoadOp(op)) {
      doAssignGroups(op, {ATTR_WS_TMALOAD});
      // if followed by a local alloc, place that in the tma load group as well
      auto desc = op->getResult(0);
      if (desc.hasOneUse()) {
        auto nextOp = *desc.user_begin();
        if (isa<triton::gpu::LocalAllocOp>(nextOp)) {
          doAssignGroups(nextOp, {ATTR_WS_TMALOAD});
        }
      }
    }

    // store and mma ops placed in mma warp group
    if (isa<triton::DotOp, triton::nvidia_gpu::WarpGroupDotOp, triton::StoreOp,
            triton::DescriptorStoreOp, triton::DescriptorScatterOp,
            triton::gpu::MemDescTransOp, triton::nvidia_gpu::TMEMAllocOp,
            triton::nvidia_gpu::TMEMLoadOp,
            triton::nvidia_gpu::MMAv5OpInterface>(op)) {
      doAssignGroups(op, {ATTR_WS_MMA});
    }

    if (numWarps.epilogue > 0) {
      if (isa<triton::StoreOp, triton::DescriptorStoreOp,
              triton::DescriptorScatterOp>(op)) {
        doAssignGroups(op, {ATTR_WS_EPILOGUE});
      } else if (isa<triton::nvidia_gpu::TMEMLoadOp>(op)) {
        // if tma_load is in the same loop as corresponding mma, it's not
        // part of the epilogue group
        auto tmemLoadOp = dyn_cast<triton::nvidia_gpu::TMEMLoadOp>(op);
        auto mmav5Op = getMMAv5Op(tmemLoadOp);
        if (mmav5Op && mmav5Op->getBlock() != tmemLoadOp->getBlock()) {
          doAssignGroups(op, {ATTR_WS_EPILOGUE});
        }
      }
    }
  }

  void doAssignGroups(Operation *op, const std::set<std::string> &groups) {
    // define groups in the module
    // FIXME: abstract this
    auto moduleOp = op->getParentOfType<ModuleOp>();
    for (auto group : groups) {
      if (group == ATTR_WS_TMALOAD) {
        mkGroup(moduleOp, ATTR_WS_TMALOAD,
                WSGroup{numWarps.epilogue + numWarps.MMA, numWarps.load});
      } else if (group == ATTR_WS_MMA) {
        mkGroup(moduleOp, ATTR_WS_MMA,
                WSGroup{numWarps.epilogue, numWarps.MMA});
      } else if (group == ATTR_WS_EPILOGUE) {
        mkGroup(moduleOp, ATTR_WS_EPILOGUE, WSGroup{0, numWarps.epilogue});
      } else {
        assert(false);
      }
    }
    // annotate op with the groups
    setGroups(op, groups);
  }
};

class NVWSAssignGroups : public NVWSAssignGroupsBase<NVWSAssignGroups> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    // initial group assignment
    if (!isManuallyGrouped(m)) {
      InitialAssignment().runOnOperation(m);
      LLVM_DEBUG({
        DBGS() << "Module after initial group assignment:\n";
        m.dump();
      });
    } else {
      LLVM_DEBUG({
        DBGS() << "Using manual initial group assignments\n";
        m.dump();
      });
      // check control flow are not assigned to groups
      m.walk([&](Operation *op) {
        if (isa<ModuleOp, FuncOp, ReturnOp, scf::IfOp, scf::ForOp,
                scf::YieldOp>(op) &&
            op->hasAttr(ATTR_WSGROUPS)) {
          op->emitError("op should not have initial group assignment");
          signalPassFailure();
        }
      });
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createNVWSAssignGroupsPass() {
  return std::make_unique<NVWSAssignGroups>();
}
