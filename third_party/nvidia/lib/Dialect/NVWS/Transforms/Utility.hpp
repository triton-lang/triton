#ifndef NVWS_TRANSFORMS_UTILITY_HPP
#define NVWS_TRANSFORMS_UTILITY_HPP

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::nvws {

template <class T> struct ThreadValue {
  std::function<Value(ImplicitLocOpBuilder &, Value, Operation *)> updateValue;
  using ValueMap = llvm::MapVector<Value /*key*/, Value /*value*/>;
  using UseSet = llvm::SetVector<Value /*key*/>;

  UseSet analyzeUseInBlock(Block *block, UseSet useSet) {
    for (auto &op : *block) {
      if (auto opT = dyn_cast<T>(op)) {
        useSet.insert(op.getOperand(0));
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        useSet = analyzeUseInBlock(forOp.getBody(), useSet);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        useSet = analyzeUseInBlock(ifOp.thenBlock(), useSet);
        if (ifOp.elseBlock())
          useSet = analyzeUseInBlock(ifOp.elseBlock(), useSet);
      }
    }
    return useSet;
  }

  void assignValueInForOp(scf::ForOp forOp, ValueMap &valueMap) {

    // find uses of key in forOp body
    auto useInBlock = analyzeUseInBlock(forOp.getBody(), {});
    if (useInBlock.empty())
      return;

    // add extra iterArgs to the forOp
    SmallVector<Value> extraIterArgs;
    SmallVector<Value *> valueRefs;
    for (auto key : useInBlock) {
      extraIterArgs.push_back(valueMap.lookup(key));
      valueRefs.push_back(&valueMap[key]);
    }

    // create new forOp with extra iterArgs
    OpBuilder builder(forOp);
    size_t nArgs = forOp.getRegionIterArgs().size();
    forOp = addIterArgsToLoop(builder, forOp, extraIterArgs);

    // update value with iterArgs in the forOp body
    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *valueRefs[idx - nArgs] = forOp.getRegionIterArgs()[idx];

    // assign value in the forOp body
    auto valueMapInBlock = assignValueInBlock(forOp.getBody(), valueMap);

    // update yieldOp to return new indexes
    SmallVector<Value> extraYieldArgs;
    for (auto key : useInBlock)
      extraYieldArgs.push_back(valueMapInBlock[key]);
    appendToForOpYield(forOp, extraYieldArgs);

    // update value with results from newForOp
    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *valueRefs[idx - nArgs] = forOp.getResult(idx);
  }

  void assignValueInIfOp(scf::IfOp ifOp, ValueMap &valueMap) {

    // find uses of key in then-block
    auto useInBlock = analyzeUseInBlock(ifOp.thenBlock(), {});
    if (useInBlock.empty())
      return;

    // find uses of key in else-block
    useInBlock = ifOp.elseBlock()
                     ? analyzeUseInBlock(ifOp.elseBlock(), useInBlock)
                     : useInBlock;

    // add extra results to the ifOp
    SmallVector<Type> extraIfResults;
    SmallVector<Value *> valueRefs;
    for (auto key : useInBlock) {
      extraIfResults.push_back(valueMap.lookup(key).getType());
      valueRefs.push_back(&valueMap[key]);
    }

    // create new ifOp with extra results
    OpBuilder builder(ifOp);
    size_t nArgs = ifOp.getResults().size();
    auto newIfOp = replaceIfOpWithNewSignature(builder, ifOp, extraIfResults);

    // assign value in then-body
    auto valueMapInThenBlock =
        assignValueInBlock(newIfOp.thenBlock(), valueMap);

    // assign value in else-body
    auto valueMapInElseBlock =
        ifOp.elseBlock() ? assignValueInBlock(newIfOp.elseBlock(), valueMap)
                         : valueMap;

    // update yieldOp to return new indexes
    auto thenYieldOp = newIfOp.thenYield();
    auto elseYieldOp = newIfOp.elseYield();
    // insert new indexes to the yieldOp
    for (auto key : useInBlock) {
      thenYieldOp->insertOperands(thenYieldOp.getNumOperands(),
                                  valueMapInThenBlock[key]);
      elseYieldOp->insertOperands(elseYieldOp.getNumOperands(),
                                  valueMapInElseBlock[key]);
    }
    ifOp.erase();

    // update value with results from newIfOp
    for (size_t idx = nArgs; idx < newIfOp.getResults().size(); ++idx)
      *valueRefs[idx - nArgs] = newIfOp.getResult(idx);
  }

  ValueMap assignValueInBlock(Block *block, ValueMap valueMap) {
    for (auto &op : llvm::make_early_inc_range(*block)) {
      if (auto opT = dyn_cast<T>(op)) {
        ImplicitLocOpBuilder b(op.getLoc(), &op);
        b.setInsertionPointAfter(&op);
        auto value = valueMap.lookup(op.getOperand(0));
        valueMap[op.getOperand(0)] = updateValue(b, value, &op);
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        assignValueInForOp(forOp, valueMap);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        assignValueInIfOp(ifOp, valueMap);
      }
    }

    return valueMap;
  }

  static void
  run(triton::nvws::WarpGroupOp wgOp,
      std::function<Value(ImplicitLocOpBuilder &, Operation *)> initValue,
      std::function<Value(ImplicitLocOpBuilder &, Value, Operation *)>
          updateValue) {
    ThreadValue<T> value{updateValue};
    UseSet useSet;
    for (auto region : wgOp.getRegions()) {
      auto block = &region->getBlocks().front();
      useSet = value.analyzeUseInBlock(block, useSet);
    }

    // initialize indexes
    ValueMap valueMap;
    for (auto key : useSet) {
      ImplicitLocOpBuilder b(key.getLoc(), key.getDefiningOp());
      b.setInsertionPointAfter(key.getDefiningOp());
      valueMap[key] = initValue(b, key.getDefiningOp());
    }

    for (auto region : wgOp.getRegions()) {
      auto block = &region->getBlocks().front();
      value.assignValueInBlock(block, valueMap);
    }
  }
};
} // namespace mlir::triton::nvws

#endif // NVWS_TRANSFORMS_UTILITY_HPP
