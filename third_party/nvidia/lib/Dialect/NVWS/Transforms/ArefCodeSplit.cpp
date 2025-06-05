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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
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
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <memory>
#include <stdatomic.h>

#define GEN_PASS_CLASSES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-lower-aref"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::gpu;
using namespace triton::nvws;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

namespace {
template <class K, class V> using Map = std::map<K, V>;
template <class K> using Set = std::set<K>;
using GroupId = std::string;
using GroupMap = Map<GroupId, WSGroup>;
using GroupSet = Set<GroupId>;

ttng::WarpGroupOp createWgOp(ModuleOp mod, int barId, int startWarp,
                             int numWarps, Location loc,
                             OpBuilderWithGroup &builder) {
  auto wgOp = builder.create<ttng::WarpGroupOp>(loc, startWarp, numWarps, 1);

  // set wgOp barId attribute
  wgOp->setAttr(ATTR_WS_BARID, builder.getI32IntegerAttr(barId));
  auto &wgRegion = wgOp.getPartitionRegions()[0];
  Block *wgBlock = &wgRegion.emplaceBlock();
  OpBuilder wgBuilder = OpBuilder::atBlockEnd(wgBlock);
  auto wgRetOp = wgBuilder.create<mlir::triton::nvidia_gpu::WarpGroupReturnOp>(
      wgOp.getLoc());
  return wgOp;
}

ttng::WarpGroupOp createWgOpWithGroup(ModuleOp mod, int barId,
                                      std::string group, Location loc,
                                      OpBuilderWithGroup &builder) {
  auto wsGroup = getGroupFromSymbolRefAttr(
      mod, mlir::SymbolRefAttr::get(builder.getContext(), group));
  return createWgOp(mod, barId, wsGroup.getStartWarp(), wsGroup.getNumWarps(),
                    loc, builder);
}

Map<std::string, ttng::WarpGroupOp> createWarpGroupOps(triton::FuncOp funcOp,
                                                       GroupMap groupMap) {
  Map<std::string, ttng::WarpGroupOp> wgOps;
  auto mod = funcOp->getParentOfType<ModuleOp>();
  SmallVector<std::pair<std::string, WSGroup>> groups;
  for (auto [groupName, group] : groupMap) {
    groups.push_back({groupName, group});
  }

  // We create warp-groups in textual order with decreasing startWarp, which
  // helps with blocked-scale performance, improving it by about ~3%:
  //   3768.558 TFLOPS block_scaled_matmul_kernel_persistent_nvfp4 [M=N=K=8192]
  // vs
  //   3883.483 TFLOPS block_scaled_matmul_kernel_persistent_nvfp4 [M=N=K=8192]

  llvm::sort(groups, [](auto a, auto b) {
    return a.second.getStartWarp() > b.second.getStartWarp();
  });

  int barId = 1;
  for (auto &[groupName, _] : groups) {
    OpBuilderWithGroup builder(funcOp, groupName);
    builder.setInsertionPoint(funcOp.getBody().front().getTerminator());
    auto wgOp =
        createWgOpWithGroup(mod, barId++, groupName, funcOp.getLoc(), builder);
    wgOps.emplace(groupName, wgOp);
  }
  return wgOps;
}

// ----------------------------------------------------------------------------

struct WgBuilder {
  OpBuilderWithGroup builder;
  IRMapping mapping;
};

/* recursive clone operations in a block to a warp-specialized function,
  wsFunc this split input block with ops annotated with groups into wsFunc,
  e.g.

   ..
   op1 @gr1, @gr2
   op2 @gr2, @gr3
   op3 @gr1
   ..

   to

   wsFunc {

    @gr1 {
       ..
       op1 @gr1
       op3 @gr1
       ..
    }

    @gr2 {
       ..
      op1  @gr2
      op2  @gr2
      ..
    }

    @gr3 {
       ..
      op2  @gr3
      ..
    }
  }

*/
void cloneOpsInBlock(Block *block, triton::FuncOp wsFunc,
                     Map<GroupId, WgBuilder> &opBuilders);

void cloneForOp(scf::ForOp forOp, triton::FuncOp wsFunc,
                Map<GroupId, WgBuilder> &opBuilders) {
  Map<GroupId, scf::ForOp> newForOps;
  for (auto group : getGroups(forOp)) {

    // clone forOp
    auto &b = opBuilders.at(group);
    auto lb = b.mapping.lookup(forOp.getLowerBound());
    auto ub = b.mapping.lookup(forOp.getUpperBound());
    auto step = b.mapping.lookup(forOp.getStep());
    SmallVector<Value> initArgs;
    for (auto [idx, arg] : llvm::enumerate(forOp.getInitArgs())) {
      auto groups = getGroupsIdx(forOp, idx);
      if (groups.count(group) > 0)
        initArgs.push_back(b.mapping.lookup(arg));
    }
    auto newForOp =
        b.builder.create<scf::ForOp>(forOp.getLoc(), lb, ub, step, initArgs);
    newForOps[group] = newForOp;
    setGroups(newForOp, {group});
    b.mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

    // map the results of the forOp to the newForOp
    auto oldIterArgs = forOp.getRegionIterArgs();
    auto newIterArgs = newForOp.getRegionIterArgs();
    for (int oldIdx = 0, newIdx = 0; oldIdx < oldIterArgs.size(); ++oldIdx) {
      auto groups = getGroupsIdx(forOp, oldIdx);
      if (groups.count(group) > 0) {
        auto oldArg = oldIterArgs[oldIdx];
        auto newArg = newIterArgs[newIdx++];
        b.mapping.map(oldArg, newArg);
      }
    }

    for (int oldIdx = 0, newIdx = 0; oldIdx < forOp.getResults().size();
         ++oldIdx) {
      auto groups = getGroupsIdx(forOp, oldIdx);
      if (groups.count(group) > 0) {
        auto oldArg = forOp.getResult(oldIdx);
        auto newArg = newForOp.getResult(newIdx++);
        b.mapping.map(oldArg, newArg);
      }
    }
    // set builder insertion point to the start of the newForOp body
    b.builder.setInsertionPointToStart(newForOp.getBody());
  }
  // resursive clone ops in the forOp body
  cloneOpsInBlock(forOp.getBody(), wsFunc, opBuilders);

  for (auto [group, newForOp] : newForOps) {
    auto &b = opBuilders.at(group);
    // set insertion point after the newForOp
    b.builder.setInsertionPointAfter(newForOp);
  }
}

void cloneIfOp(scf::IfOp ifOp, triton::FuncOp wsFunc,
               Map<GroupId, WgBuilder> &opBuilders) {
  Map<GroupId, scf::IfOp> newIfOps;
  for (auto group : getGroups(ifOp)) {
    auto &b = opBuilders.at(group);

    SmallVector<Type> newIfResultTypes;
    for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
      auto groups = getGroupsIdx(ifOp, idx);
      if (groups.count(group))
        newIfResultTypes.push_back(result.getType());
    }
    auto cond = b.mapping.lookup(ifOp.getCondition());
    auto newIfOp = b.builder.create<scf::IfOp>(
        ifOp.getLoc(), newIfResultTypes, cond, ifOp.elseBlock() ? true : false);
    newIfOps[group] = newIfOp;
    setGroups(newIfOp, {group});

    // map results
    for (int oldIdx = 0, newIdx = 0; oldIdx < ifOp.getResults().size();
         ++oldIdx) {
      auto groups = getGroupsIdx(ifOp, oldIdx);
      if (groups.count(group) > 0) {
        auto oldArg = ifOp.getResult(oldIdx);
        auto newArg = newIfOp.getResult(newIdx++);
        b.mapping.map(oldArg, newArg);
      }
    }
    // map block args
    for (auto [oldArg, newArg] : llvm::zip(ifOp.thenBlock()->getArguments(),
                                           newIfOp.thenBlock()->getArguments()))
      b.mapping.map(oldArg, newArg);
    if (ifOp.elseBlock())
      for (auto [oldArg, newArg] :
           llvm::zip(ifOp.elseBlock()->getArguments(),
                     newIfOp.elseBlock()->getArguments()))
        b.mapping.map(oldArg, newArg);

    b.builder.setInsertionPointToStart(newIfOp.thenBlock());
  }

  cloneOpsInBlock(ifOp.thenBlock(), wsFunc, opBuilders);

  if (auto elseBlock = ifOp.elseBlock()) {
    for (auto [group, newIfOp] : newIfOps) {
      auto &b = opBuilders.at(group);
      b.builder.setInsertionPointToStart(newIfOp.elseBlock());
    }
    cloneOpsInBlock(elseBlock, wsFunc, opBuilders);
  }

  for (auto [group, newIfOp] : newIfOps) {
    auto &b = opBuilders.at(group);
    b.builder.setInsertionPointAfter(newIfOp);
  }
}

void cloneReduceOp(triton::ReduceOp reduceOp, triton::FuncOp wsFunc,
                   Map<GroupId, WgBuilder> &opBuilders) {
  Map<GroupId, triton::ReduceOp> newReduceOps;
  for (auto group : getGroups(reduceOp)) {
    auto &b = opBuilders.at(group);

    // map arguments
    SmallVector<Value> srcs;
    for (auto src : reduceOp.getSrcs())
      srcs.push_back(b.mapping.lookup(src));
    auto axis = reduceOp.getAxis();

    auto newReduceOp =
        b.builder.create<triton::ReduceOp>(reduceOp.getLoc(), srcs, axis);
    newReduceOps[group] = newReduceOp;
    setGroups(newReduceOp, {group});

    for (auto [oldResult, newResult] :
         llvm::zip(reduceOp.getResults(), newReduceOp.getResults()))
      b.mapping.map(oldResult, newResult);

    auto &region = newReduceOp.getRegion();
    Block *block = &region.emplaceBlock();
    for (auto args : reduceOp.getRegion().getBlocks().front().getArguments()) {
      auto newArg = block->addArgument(args.getType(), args.getLoc());
      b.mapping.map(args, newArg);
    }

    b.builder.setInsertionPointToStart(block);
  }
  cloneOpsInBlock(reduceOp.getBody(), wsFunc, opBuilders);
  for (auto [group, newReduceOp] : newReduceOps) {
    auto &b = opBuilders.at(group);
    b.builder.setInsertionPointAfter(newReduceOp);
  }
}

void cloneOpsInBlock(Block *block, triton::FuncOp wsFunc,
                     Map<GroupId, WgBuilder> &opBuilders) {
  OpBuilder topBuilder(wsFunc);
  topBuilder.setInsertionPointToStart(&wsFunc.getBody().front());
  IRMapping topMapping;
  for (auto &op_ : *block) {
    auto op = &op_;

    if (isa<triton::ReturnOp>(op))
      continue;

    auto producerGroups = getGroups(op);

    if (isa<LocalAllocOp, ttng::TMEMAllocOp, ttng::ArefCreateOp>(op) &&
        producerGroups.empty()) {
      // allocs w/o group annotation are reserved for aref_creation
      // and cloned to the top -evel
      auto clonedOp = topBuilder.clone(*op, topMapping);
      for (auto [oldResult, newResult] :
           llvm::zip(op->getResults(), clonedOp->getResults()))
        for (auto &[_, b] : opBuilders) {
          b.mapping.map(oldResult, newResult);
          topMapping.map(oldResult, newResult);
        }
      continue;
    }

    if (!isa<scf::YieldOp, ttng::ArefPhiOp>(op)) {
      // only Yield  & ArefPhi doesn't have annotation
      assert(!producerGroups.empty());
    }

    if (auto arefPhiOp = dyn_cast<ttng::ArefPhiOp>(op)) {
      auto producedVar = arefPhiOp.getLocal();
      auto consumedVar = arefPhiOp.getRemote();

      auto getOpGroups = [](Value v) {
        auto op = v.getDefiningOp();
        if (isa<scf::ForOp, scf::IfOp>(op)) {
          auto resultIdx = -1;
          for (auto [idx, result] : llvm::enumerate(op->getResults())) {
            if (result == v) {
              resultIdx = idx;
              break;
            }
          }
          assert(resultIdx != -1);

          return getGroupsIdx(op, resultIdx);
        } else {
          return getGroups(op);
        }
      };

      auto producerGroups = getOpGroups(producedVar);
      auto consumerGroups = getOpGroups(consumedVar);

      for (auto group : producerGroups) {
        auto &b = opBuilders.at(group);
        b.mapping.map(arefPhiOp, b.mapping.lookup(producedVar));
      }
      for (auto group : consumerGroups) {
        if (producerGroups.count(group) == 0) {
          auto &b = opBuilders.at(group);
          b.mapping.map(arefPhiOp, b.mapping.lookup(consumedVar));
        }
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      cloneForOp(forOp, wsFunc, opBuilders);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      cloneIfOp(ifOp, wsFunc, opBuilders);
    } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      cloneReduceOp(reduceOp, wsFunc, opBuilders);
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {

      // yield %a0, %a1, .. , %an
      // each group yields its own operand
      Map<GroupId, SmallVector<Value>> yieldOperands;
      for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {
        auto parentOp = yieldOp->getParentOp();
        for (auto groupName : getGroupsIdx(parentOp, idx))
          yieldOperands[groupName].push_back(
              opBuilders.at(groupName).mapping.lookup(operand));
      }

      for (auto &[groupName, operands] : yieldOperands) {
        auto &b = opBuilders.at(groupName);
        b.builder.create<scf::YieldOp>(op->getLoc(), operands);
      }

    } else {

      // all remaining ops are expected to be regionless
      assert(op->getNumRegions() == 0);

      // clone op into each producer group
      for (auto producerGroup : producerGroups) {
        auto &b = opBuilders.at(producerGroup);
        auto newOp = b.builder.clone(*op, b.mapping);
        setGroups(newOp, {producerGroup});
        for (auto [oldResult, newResult] :
             llvm::zip(op->getResults(), newOp->getResults()))
          b.mapping.map(oldResult, newResult);
      }
    }
  }
}

void splitFunc(triton::FuncOp &funcOp, GroupMap groups) {
  // clone function
  OpBuilder builder(funcOp);
  builder.setInsertionPoint(funcOp);
  Operation *op = funcOp;
  auto wsFuncOp = builder.clone(*op);
  auto wsFunc = dyn_cast<triton::FuncOp>(wsFuncOp);
  assert(wsFunc && "failed to clone function");
  // remove contect of wsFunc
  wsFunc.getBody().front().clear();
  // insert tt.return at the end of the function
  builder.setInsertionPointToEnd(&wsFunc.getBody().front());
  builder.create<mlir::triton::ReturnOp>(wsFunc.getLoc());
  IRMapping mapping;
  auto oldArgs = funcOp.getArguments();
  auto newArgs = wsFunc.getArguments();

  auto oldName = funcOp.getName().str() + "__OLD";
  funcOp.setSymName(oldName);

  auto wgOps = createWarpGroupOps(wsFunc, groups);
  Map<GroupId, WgBuilder> opBuilders;
  for (auto &[groupName, wgOp] : wgOps) {
    OpBuilderWithGroup builder(wgOp, groupName);
    auto block = &wgOp.getRegions().front()->front();
    builder.setInsertionPointToStart(block);

    IRMapping mapping;
    for (auto [oldArg, newArg] : llvm::zip(oldArgs, newArgs))
      mapping.map(oldArg, newArg);
    opBuilders.emplace(groupName, WgBuilder{builder, mapping});
  }

  cloneOpsInBlock(&funcOp.getBody().front(), wsFunc, opBuilders);

  funcOp.erase();
  funcOp = wsFunc;
}

void insertInitBarrier(tt::FuncOp funcOp) {
  SmallVector<ttng::WarpGroupOp> wgOps;
  funcOp.walk([&](ttng::WarpGroupOp wgOp) { wgOps.push_back(wgOp); });
  auto firstWg = llvm::min_element(
      wgOps, [](auto a, auto b) { return a->isBeforeInBlock(b); });
  OpBuilder builder(*firstWg);
  auto barrier = builder.create<NVVM::Barrier0Op>(funcOp.getLoc());
  barrier->setAttr(ATTR_WS_INIT_BARRIER_SYNC, builder.getUnitAttr());
}

} // namespace

class NVWSArefCodeSplit : public NVWSArefCodeSplitBase<NVWSArefCodeSplit> {
public:
  void runOnFuncOp(triton::FuncOp funcOp, GroupMap groups) {
    splitFunc(funcOp, groups);
    LLVM_DEBUG({ DBGS() << "after::splitFunc:\n" << funcOp << "\n"; });

    insertInitBarrier(funcOp);
    LLVM_DEBUG({ DBGS() << "after::insertInitBarrier:\n" << funcOp << "\n"; });
  }

  void runOnOperation() override {
    auto mod = getOperation();
    auto groups = collectGroups(mod);
    if (groups.empty())
      return;

    mod->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp, groups); });
  }
}; // namespace

} // namespace

std::unique_ptr<Pass> mlir::createNVWSArefCodeSplitPass() {
  return std::make_unique<NVWSArefCodeSplit>();
}
