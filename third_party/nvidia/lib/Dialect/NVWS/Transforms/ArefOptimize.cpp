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
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Casting.h"

#include <memory>

// #define GEN_PASS_CLASSES
// #include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define GEN_PASS_CLASSES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-optimize"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::gpu;
using namespace triton::nvws;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

bool shouldReallocRegisters(ModuleOp mod) { return isHopper(mod); }

void hoistConstTMEMAllocFromWgs(tt::FuncOp funcOp) {
  SmallVector<ttng::WarpGroupOp> wgOps;
  funcOp.walk([&](ttng::WarpGroupOp wgOp) { wgOps.push_back(wgOp); });

  SmallVector<ttng::TMEMAllocOp> tmemAllocs;
  for (auto wgOp : wgOps) {
    wgOp.walk(
        [&](ttng::TMEMAllocOp tmemAlloc) { tmemAllocs.push_back(tmemAlloc); });
  }

  auto wgFirst = *llvm::min_element(
      wgOps, [](ttng::WarpGroupOp wgOp1, ttng::WarpGroupOp wgOp2) {
        return wgOp1->isBeforeInBlock(wgOp2);
      });

  for (auto tmemAlloc : tmemAllocs) {
    if (auto src = tmemAlloc.getSrc())
      if (auto constOp = dyn_cast<arith::ConstantOp>(src.getDefiningOp())) {
        constOp->moveBefore(wgFirst);
        tmemAlloc->moveBefore(wgFirst);
      }
  }
}

template <typename EnterOp, typename ExitOp>
void createCombinedArefOps(SmallVector<EnterOp> &enterOps,
                           SmallVector<ExitOp> &exitOps,
                           ttng::ArefCreateOp aref, OpBuilder &builder) {
  auto firstEnter = *llvm::min_element(enterOps, [](auto a, auto b) {
    assert(a->getBlock() == b->getBlock());
    return a->isBeforeInBlock(b);
  });

  auto lastExit = *llvm::max_element(exitOps, [](auto a, auto b) {
    assert(a->getBlock() == b->getBlock());
    return a->isBeforeInBlock(b);
  });

  SmallVector<Type> arefEnterBuffers, arefEnterTokens;
  for (auto enterOp : enterOps) {
    arefEnterBuffers.push_back(enterOp.getBuffers()[0].getType());
    arefEnterTokens.push_back(enterOp.getTokens()[0].getType());
    assert(isa<NoneType>(arefEnterTokens.back()));
  }

  llvm::SmallSetVector<Attribute, 5> opAttrsSet;
  for (Operation *exitOp : exitOps) {
    auto arrayAttrs =
        cast<ttng::ArefExitOpInterface>(exitOp).getAssociatedOpAttrs();
    opAttrsSet.insert(arrayAttrs[0]);
  }
  llvm::SmallVector<Attribute> producersOrConsumers(opAttrsSet.begin(),
                                                    opAttrsSet.end());

  builder.setInsertionPoint(firstEnter);
  auto zero =
      mkConstant(builder, firstEnter->getLoc(), 0, 32, getGroups(firstEnter));
  auto enter = builder.create<EnterOp>(firstEnter->getLoc(), arefEnterBuffers,
                                       arefEnterTokens, aref, zero);
  builder.setInsertionPoint(lastExit);
  auto exit =
      builder.create<ExitOp>(lastExit->getLoc(), aref, zero,
                             builder.getArrayAttr(producersOrConsumers));

  for (auto [idx, enterOp] : llvm::enumerate(enterOps))
    enterOp.getBuffers()[0].replaceAllUsesWith(enter.getBuffers()[idx]);

  for (auto op : SmallVector<Operation *>{enter, exit}) {
    op->setAttr("aref_tag", firstEnter->getAttr("aref_tag"));
    setGroups(op, getGroups(firstEnter));
  }
}

void combineArefs(triton::FuncOp funcOp) {
  // this subpass will combine arefs into a single one if aref are
  // used by the same op in the same block:

  // %buf_a = alloc(); %aref_a = aref_create %buf_a
  // %buf_b = alloc(); %aref_b = aref_create %buf_b

  // %a = aref_get.enter %aref_a
  // %b = aref_get.enter %aref_b
  //   .. = op .. %a, .. , %b ..
  // aref_get.exit %aref_a
  // aref_get.exit %aref_B

  // %a = aref_put.neter %aref_a
  //  store .. %a
  // aref_put.exit %aref_a
  // %b = aref_put.neter %aref_b
  //  store .. %b
  // aref_put.exit %aref_b

  // becomes

  // %buf_a = alloc(); %buf_b = alloc(); %aref_ab = aref_create %buf_a,
  // %buf_b

  // %a = aref_get.enter %aref_ab
  // %b = aref_get.enter %aref_ab
  //   .. = op .. %a, .. , %b ..
  // aref_get.exit %aref_ab

  // %a,5b = aref_put.enter %aref_ab
  //  store .. %a
  //  store .. %b
  // aref_put.exit %aref_ab

  // for now this happens at MMA sites, so we just visit MMA ops, generic
  // algorithm can be implemented at a later time

  std::function<ttng::ArefCreateOp(Value)> findAref =
      [&](Value opnd) -> ttng::ArefCreateOp {
    if (!opnd)
      return {};
    if (auto op = opnd.getDefiningOp()) {
      if (auto enterOp = dyn_cast<ttng::ArefGetEnterOp>(op)) {
        return cast<ttng::ArefCreateOp>(enterOp.getAref().getDefiningOp());
      } else {
        for (auto operand : op->getOperands())
          if (auto aref = findAref(operand))
            return aref;
      }
    }
    return {};
  };

  SmallVector<SmallVector<ttng::ArefCreateOp>> arefsToFuse;
  funcOp.walk([&](Operation *op) {
    Value Aopnd, Bopnd, AScaleOpnd, BScaleOpnd;
    if (auto wgmma = dyn_cast<ttng::WarpGroupDotOp>(op)) {
      Aopnd = wgmma.getA();
      Bopnd = wgmma.getB();
    } else if (auto mmav5 = dyn_cast<ttng::TCGen5MMAOp>(op)) {
      Aopnd = mmav5.getA();
      Bopnd = mmav5.getB();
    } else if (auto mmav5scaled = dyn_cast<ttng::TCGen5MMAScaledOp>(op)) {
      Aopnd = mmav5scaled.getA();
      Bopnd = mmav5scaled.getB();
      AScaleOpnd = mmav5scaled.getAScale();
      BScaleOpnd = mmav5scaled.getBScale();
    } else {
      return WalkResult::advance();
    }

    auto usedInTheSameBlock = [](ttng::ArefCreateOp arefA,
                                 ttng::ArefCreateOp arefB) {
      Block *putABlock, *getABlock;
      Block *putBBlock, *getBBlock;
      for (auto user : arefA->getUsers()) {
        if (isa<ttng::ArefPutEnterOp>(user)) {
          putABlock = user->getBlock();
        } else if (isa<ttng::ArefGetEnterOp>(user)) {
          getABlock = user->getBlock();
        }
      }

      for (auto user : arefB->getUsers()) {
        if (isa<ttng::ArefPutEnterOp>(user)) {
          putBBlock = user->getBlock();
        } else if (isa<ttng::ArefGetEnterOp>(user)) {
          getBBlock = user->getBlock();
        }
      }

      return putABlock == putBBlock && getABlock == getBBlock;
    };

    auto usedOnce = [](ttng::ArefCreateOp aref) {
      return llvm::count_if(aref->getUsers(), [](auto) { return true; }) == 4;
    };

    // verify that all put/gets are in the same BB
    auto arefA = findAref(Aopnd);
    auto arefB = findAref(Bopnd);
    SmallVector<ttng::ArefCreateOp> arefs;
    if (arefA && arefB && usedOnce(arefA) && usedOnce(arefB) &&
        usedInTheSameBlock(arefA, arefB)) {
      arefs.push_back(arefA);
      arefs.push_back(arefB);
    } else {
      return WalkResult::advance();
    }

    auto arefAScale = findAref(AScaleOpnd);
    if (arefAScale &&
        isa<ttg::SharedMemorySpaceAttr>(
            cast<MemDescType>(AScaleOpnd.getType()).getMemorySpace()) &&
        usedOnce(arefAScale) && usedInTheSameBlock(arefAScale, arefA))
      arefs.push_back(arefAScale);

    auto arefBScale = findAref(BScaleOpnd);
    if (arefBScale &&
        isa<ttg::SharedMemorySpaceAttr>(
            cast<MemDescType>(BScaleOpnd.getType()).getMemorySpace()) &&
        usedOnce(arefBScale) && usedInTheSameBlock(arefBScale, arefA))
      arefs.push_back(arefBScale);

    if (!arefs.empty())
      arefsToFuse.push_back(arefs);
    return WalkResult::advance();
  });

  //   now fuse arefs
  for (auto arefs : arefsToFuse) {
    SmallVector<ttng::ArefPutEnterOp> putEnterOps;
    SmallVector<ttng::ArefPutExitOp> putExitOps;
    SmallVector<ttng::ArefGetEnterOp> getEnterOps;
    SmallVector<ttng::ArefGetExitOp> getExitOps;
    for (auto aref : arefs) {
      for (auto user : aref->getUsers()) {
        if (auto putEnterOp = dyn_cast<ttng::ArefPutEnterOp>(user)) {
          putEnterOps.push_back(putEnterOp);
        } else if (auto putExitOp = dyn_cast<ttng::ArefPutExitOp>(user)) {
          putExitOps.push_back(putExitOp);
        } else if (auto getEnterOp = dyn_cast<ttng::ArefGetEnterOp>(user)) {
          getEnterOps.push_back(getEnterOp);
        } else if (auto getExitOp = dyn_cast<ttng::ArefGetExitOp>(user)) {
          getExitOps.push_back(getExitOp);
        }
      }
    }

    // set insertion point at the last aref_create
    SmallVector<Type> arefBufTypes;
    SmallVector<Value> arefBufs;
    for (auto aref : arefs) {
      arefBufTypes.push_back(aref.getOperands()[0].getType());
      arefBufs.push_back(aref.getOperands()[0]);
    }
    auto lastAref = *llvm::max_element(arefs, [](auto a, auto b) {
      assert(a->getBlock() == b->getBlock());
      return a->isBeforeInBlock(b);
    });
    OpBuilder builder(lastAref);
    auto arefTy = triton::nvidia_gpu::ArefType::get(
        builder.getContext(),
        ttg::TypeArrayAttr::get(builder.getContext(), arefBufTypes));
    auto aref = builder.create<ttng::ArefCreateOp>(lastAref->getLoc(), arefTy,
                                                   arefBufs);

    createCombinedArefOps(putEnterOps, putExitOps, aref, builder);
    createCombinedArefOps(getEnterOps, getExitOps, aref, builder);

    for (auto putEnterOp : putEnterOps)
      putEnterOp->erase();
    for (auto putExitOp : putExitOps)
      putExitOp->erase();
    for (auto getEnterOp : getEnterOps)
      getEnterOp->erase();
    for (auto getExitOp : getExitOps)
      getExitOp->erase();
    for (auto aref : arefs)
      aref->erase();
  }
}

void reuseArefs(triton::FuncOp funcOp) {
  /* the ArefyCode pass will create a new aref per communication, which
     creates an opportunity to optimize arefs use, by reusing some, e.g. in
     case of causal FA, we have two loops

      %arefK1 = aref_create ..
      %arefV1 = aref_create..
      %arefK2 = aref_create ..
      %arefV2 = aref_create..
      ...
      for .. {
        .. use %arefK1 & %arefV1
      }
      ..
      for .. {
        .. use %arefK2 & %arefV2
      }

      To reduce memory usage, we can reuse %arefK1 & %arefV1 in the second

      %arefK1 = aref_create ..
      %arefV1 = aref_create..
      ..
      for .. {
        .. use %arefK1 & %arefV1
      }
      ..
      for .. {
        .. use %arefK1 & %arefV1
      }
      ..
  */

  // for now apply this only to FA case as we have some utility
  // to identify FA loops

  SmallVector<scf::ForOp> fmhaMathForOps;
  funcOp.walk([&](scf::ForOp forOp) {
    if (isFMHAMathLoop(forOp))
      fmhaMathForOps.push_back(forOp);
  });
  LLVM_DEBUG(
      { DBGS() << "fmhaMathForOps.size() " << fmhaMathForOps.size() << "\n"; });
  if (fmhaMathForOps.size() == 2) {
    // find arefPutEnterOp from the first loop
    DenseMap<Value, Value> loadToArefMap[2];
    for (int idx = 0; idx < 2; ++idx)
      fmhaMathForOps[idx].walk([&](ttng::ArefGetEnterOp getEnterOp) {
        if (getEnterOp.getAref().getDefiningOp()->hasAttr("aref_mma_sync"))
          return;

        ttng::ArefPutEnterOp putEnterOp;
        for (auto user : getEnterOp.getAref().getUsers())
          if (auto enterOp = dyn_cast<ttng::ArefPutEnterOp>(user))
            putEnterOp = enterOp;
        assert(putEnterOp);
        SmallVector<Operation *> users(putEnterOp->getUsers().begin(),
                                       putEnterOp->getUsers().end());
        assert(users.size() == 1);
        auto user = users[0];
        if (auto localStore = dyn_cast<LocalStoreOp>(user)) {
          auto src = localStore.getSrc().getDefiningOp();
          if (auto load = dyn_cast<tt::DescriptorLoadOp>(src))
            loadToArefMap[idx][load.getDesc()] = putEnterOp.getAref();
        } else if (auto load = dyn_cast<triton::nvws::DescriptorLoadOp>(user)) {
          loadToArefMap[idx][load.getDesc()] = putEnterOp.getAref();
        } else {
          llvm_unreachable("unsupported user");
        }
      });

    // now ruse arefs from first loop in second loop
    for (auto [desc, aref] : loadToArefMap[1]) {
      aref.replaceAllUsesWith(loadToArefMap[0].at(desc));
      SmallVector<Operation *> opsToErase;
      for (auto opnd : aref.getDefiningOp()->getOperands())
        opsToErase.push_back(opnd.getDefiningOp());
      aref.getDefiningOp()->erase();
      for (auto op : opsToErase)
        op->erase();
    }
  }

  // Note: explore a general algorithm to reuse arefs
}

void optimizeNumWarps(tt::FuncOp funcOp) {
  if (isManuallyGrouped(funcOp)) {
    // skip if program has manually assigned groups
    return;
  }

  SmallVector<ttng::WarpGroupOp> wgOps;
  funcOp.walk([&](ttng::WarpGroupOp wgOp) { wgOps.push_back(wgOp); });

  if (isHopper(funcOp->getParentOfType<ModuleOp>())) {
    // For backward compatibility, keep num_warps = 4 on Hopper.
    return;
  }

  for (auto wgOp : wgOps) {
    // Check if the load group has cpasync in it. In that case, we clearly
    // cannot make it a 1-warp group. Moreover, for some reason, the start id of
    // the cpasync warp group must be aligned to a multiple of 4 for
    // correctness. So the MMA group needs to have 4 warps as well (the load
    // group follows the MMA one).
    // TODO: Root cause and fix this.
    if (isOpInGroup(wgOp, ATTR_WS_TMALOAD)) {
      bool hasCpAsync = false;
      for (auto &block : wgOp.getRegion(0)) {
        block.walk([&](AsyncCopyGlobalToLocalOp op) { hasCpAsync = true; });
      }
      if (hasCpAsync) {
        auto mod = wgOp->getParentOfType<ModuleOp>();
        auto numWarpsAttr =
            mlir::cast<mlir::IntegerAttr>(mod->getAttr(AttrNumWarpsName));
        assert(numWarpsAttr.getInt() == 4 &&
               "only num_warps = 4 supported when using cpasync.");
        return;
      }
    }
  }

  auto groups = collectGroups(funcOp->getParentOfType<ModuleOp>());

  llvm::sort(wgOps, [](ttng::WarpGroupOp wgOp1, ttng::WarpGroupOp wgOp2) {
    return wgOp1.getStartWarp() < wgOp2.getStartWarp();
  });

  OpBuilder builder(funcOp.getContext());
  int startWarp = 0;
  for (auto &wgOp : wgOps) {
    bool hasSIMTOp = false;

    for (auto &block : wgOp.getRegion(0)) {
      block.walk([&](Operation *op) {
        if (isa<DescriptorOpInterface>(op)) {
          return WalkResult::advance();
        }
        if (auto localStore = dyn_cast<LocalStoreOp>(op)) {
          auto src = localStore.getSrc().getDefiningOp();
          if (isa<DescriptorOpInterface>(src)) {
            return WalkResult::advance();
          }
        }
        if (auto localAlloc = dyn_cast<LocalAllocOp>(op)) {
          if (auto src = localAlloc.getSrc()) {
            if (isa<DescriptorOpInterface>(src.getDefiningOp())) {
              return WalkResult::advance();
            }
          }
        }
        if (isSIMTOp(op)) {
          hasSIMTOp = true;
        }
        return WalkResult::advance();
      });
    }

    int newNumWarps = wgOp.getNumWarps();
    if (isOpInGroup(wgOp, ATTR_WS_EPILOGUE)) {
      // check eplogue group starts at warp 0
      // TODO: really we need to check %4, but requiring 0 for now
      assert(wgOp.getStartWarp() == 0);
    }

    if (!hasSIMTOp) {
      if (!isOpInGroup(wgOp, ATTR_WS_TMALOAD) || wgOp.getStartWarp() != 0) {
        // If TMA group has startWarp = 0, we cannot reduce its num_warps
        // to 1 Possibly related, when there is an epilogue group and both
        // MMA and TMA groups get num_warps = 1, the TMA group must come
        // after MMA.
        // TODO: Investigate why
        newNumWarps = 1;
      }
    }

    auto group = getGroup(wgOp);
    auto groupInfo = groups[group];

    wgOp.setStartWarp(startWarp);
    wgOp.setNumWarps(newNumWarps);
    setGroupAttribute(funcOp->getParentOfType<ModuleOp>(), group,
                      WSGroup(startWarp, newNumWarps, groupInfo.getRegCount()));

    startWarp += newNumWarps;
  }
}

void allocateRegisters(triton::FuncOp funcOp) {
  if (shouldReallocRegisters(funcOp->getParentOfType<ModuleOp>())) {
    funcOp.walk([&](ttng::WarpGroupOp wgOp) {
      OpBuilder builder(wgOp);
      builder.setInsertionPointToStart(&wgOp.getRegion(0).front());
      auto groups = getGroups(wgOp);
      for (auto group : groups) {
        if (group == ATTR_WS_TMALOAD) {
          wgOp.setRegCount(40);
        } else if (group == ATTR_WS_MMA) {
          wgOp.setRegCount(232);
        }
      }
    });
  }
}

class NVWSArefOptimize : public NVWSArefOptimizeBase<NVWSArefOptimize> {
public:
  void runOnFunction(tt::FuncOp funcOp) {
    hoistConstTMEMAllocFromWgs(funcOp);
    LLVM_DEBUG({ DBGS() << "after::hoistTMEMAllocs:\n" << funcOp << "\n"; });

    combineArefs(funcOp);
    LLVM_DEBUG({ DBGS() << "after::combineArefs:\n" << funcOp << "\n"; });

    reuseArefs(funcOp);
    LLVM_DEBUG({ DBGS() << "after::reuseArefs:\n" << funcOp << "\n"; });

    optimizeNumWarps(funcOp);
    LLVM_DEBUG({ DBGS() << "after::optimizeNumWarps:\n" << funcOp << "\n"; });

    allocateRegisters(funcOp);
    LLVM_DEBUG({ DBGS() << "after::allocateRegisters:\n" << funcOp << "\n"; });
  }

  void runOnOperation() override {
    auto mod = getOperation();
    mod.walk([&](tt::FuncOp funcOp) { runOnFunction(funcOp); });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createNVWSArefOptimizePass() {
  return std::make_unique<NVWSArefOptimize>();
}
