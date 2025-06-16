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

#include "nvidia/include/Dialect/NVWS/Transforms/Utility.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include <optional>

namespace mlir {
namespace triton {
namespace nvws {

namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

bool isManuallyGrouped(ModuleOp module) {
  auto attr = module->getAttr(ATTR_WS_MANUAL);
  return attr && cast<BoolAttr>(attr).getValue();
}

bool isManuallyGrouped(Operation *op) {
  return isManuallyGrouped(op->getParentOfType<ModuleOp>());
}

namespace {

std::set<std::string> getGroups(Operation *op, const std::string &attrName) {
  std::set<std::string> groups;
  if (op->hasAttr(attrName)) {
    for (auto attr : cast<ArrayAttr>(op->getAttr(attrName))) {
      groups.insert(cast<SymbolRefAttr>(attr).getRootReference().str());
    }
  }
  return groups;
}

void setGroups(Operation *op, const std::string &attrName,
               const std::set<std::string> &groups) {
  SmallVector<Attribute, 4> attrs;
  for (auto group : groups) {
    attrs.push_back(SymbolRefAttr::get(op->getContext(), group));
  }
  op->setAttr(attrName, ArrayAttr::get(op->getContext(), attrs));
}

void addGroups(Operation *op, const std::string &attrName,
               const std::set<std::string> &groups) {
  if (op->hasAttr(attrName)) {
    auto newGroups = getGroups(op, attrName);
    newGroups.insert(groups.begin(), groups.end());
    setGroups(op, attrName, newGroups);
  } else {
    setGroups(op, attrName, groups);
  }
}

} // namespace

std::set<std::string> getGroups(Operation *op) {
  return getGroups(op, ATTR_WS_GROUPS);
}

std::set<std::string> getGroups(OpResult result) {
  return getGroups(result.getOwner(),
                   std::string(ATTR_WS_GROUPS) + "." +
                       std::to_string(result.getResultNumber()));
}

std::set<std::string> getGroupsIdx(Operation *op, int idx) {
  return getGroups(op, std::string(ATTR_WS_GROUPS) + "." + std::to_string(idx));
}

void setGroups(Operation *op, const std::set<std::string> &groups) {
  setGroups(op, ATTR_WS_GROUPS, groups);
}

void setGroups(OpResult result, const std::set<std::string> &groups) {
  setGroups(result.getOwner(),
            std::string(ATTR_WS_GROUPS) + "." +
                std::to_string(result.getResultNumber()),
            groups);
}
void setGroupsIdx(Operation *op, int idx, const std::set<std::string> &groups) {
  setGroups(op, std::string(ATTR_WS_GROUPS) + "." + std::to_string(idx),
            groups);
}

void addGroups(Operation *op, const std::set<std::string> &groups) {
  addGroups(op, ATTR_WS_GROUPS, groups);
}

void addGroups(OpResult result, const std::set<std::string> &groups) {
  auto op = result.getOwner();
  auto attrName = std::string(ATTR_WS_GROUPS) + "." +
                  std::to_string(result.getResultNumber());
  addGroups(op, attrName, groups);
}
void addGroupsIdx(Operation *op, int idx, const std::set<std::string> &groups) {
  auto attrName = std::string(ATTR_WS_GROUPS) + "." + std::to_string(idx);
  addGroups(op, attrName, groups);
}

void copyGroups(Operation *from_op, Operation *to_op) {
  if (from_op->hasAttr(ATTR_WS_GROUPS)) {
    setGroups(to_op, ATTR_WS_GROUPS, getGroups(from_op, ATTR_WS_GROUPS));
  }
}

SmallVector<ttng::WarpGroupOp> findWarpGroupOps(ModuleOp m) {
  SmallVector<ttng::WarpGroupOp> wgOps;
  m.walk([&](ttng::WarpGroupOp wgOp) { wgOps.push_back(wgOp); });
  return wgOps;
}

std::string getGroup(nvidia_gpu::WarpGroupOp wgOp) {
  auto groups = getGroups(wgOp);
  assert(groups.size() == 1);
  return *groups.begin();
}

WSGroup getGroupFromAttribute(Attribute attr) {
  assert(attr);
  auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(attr);
  assert(dictAttr);
  int startWarp =
      (int)mlir::cast<IntegerAttr>(dictAttr.get(ATTR_WS_START_WARP)).getInt();
  int numWarps =
      (int)mlir::cast<IntegerAttr>(dictAttr.get(ATTR_WS_NUM_WARPS)).getInt();
  int regCount = 0;
  if (dictAttr.contains(ATTR_WS_REG_COUNT)) {
    regCount =
        (int)mlir::cast<IntegerAttr>(dictAttr.get(ATTR_WS_REG_COUNT)).getInt();
  }
  return WSGroup(startWarp, numWarps, regCount);
}

WSGroup getGroupFromSymbolRefAttr(ModuleOp mod, SymbolRefAttr refAttr) {
  mlir::Attribute attr = mod->getAttr(refAttr.getRootReference());
  assert(attr);
  return getGroupFromAttribute(attr);
}

bool isOpInGroup(Operation *op, const std::string &group) {
  if (!op->hasAttr(ATTR_WS_GROUPS)) {
    return false;
  }
  auto groups = getGroups(op, ATTR_WS_GROUPS);
  return groups.count(group) > 0;
}

bool isResultInGroup(Value value, const std::string &group) {
  auto result = cast<OpResult>(value);
  auto op = result.getOwner();
  auto attrName = std::string(ATTR_WS_GROUPS) + "." +
                  std::to_string(result.getResultNumber());
  if (!op->hasAttr(attrName)) {
    return false;
  }
  return getGroups(op, attrName).count(group) > 0;
}

void setGroupAttribute(ModuleOp moduleOp, const std::string &name,
                       WSGroup group) {
  assert(!moduleOp->hasAttr(ATTR_WS_GROUPS));
  OpBuilder builder(moduleOp.getContext());
  auto attr_pair = std::vector<NamedAttribute>{
      {builder.getStringAttr(ATTR_WS_START_WARP),
       builder.getI32IntegerAttr(group.getStartWarp())},
      {builder.getStringAttr(ATTR_WS_NUM_WARPS),
       builder.getI32IntegerAttr(group.getNumWarps())}};
  if (group.hasRegCount()) {
    attr_pair.push_back({builder.getStringAttr(ATTR_WS_REG_COUNT),
                         builder.getI32IntegerAttr(group.getRegCount())});
  }
  NamedAttrList attrList(attr_pair);
  auto dictAttr = DictionaryAttr::get(moduleOp.getContext(), attrList);
  moduleOp->setAttr(name, dictAttr);
}

SymbolRefAttr mkGroup(ModuleOp moduleOp, const std::string &name,
                      WSGroup group) {
  if (!moduleOp->hasAttr(name)) {
    // if module has group set, just return reference
    setGroupAttribute(moduleOp, name, group);
  }
  auto context = moduleOp.getContext();
  auto attrRef = SymbolRefAttr::get(context, name);
  return attrRef;
}

// Simplified from the Meta code in WSCodePartition.cpp
// Compute ((upperBound - lowerBound) + forOpStep - 1) / forOpStep
Value getLoopNumIter(Value lb, Value ub, Value step, Location loc,
                     OpBuilder &builder) {
  Value numSteps = builder.create<arith::SubIOp>(loc, ub, lb);
  numSteps = builder.create<arith::AddIOp>(loc, numSteps, step);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  numSteps = builder.create<arith::SubIOp>(loc, numSteps, one);
  return builder.create<arith::DivSIOp>(loc, numSteps, step);
}

Value getLoopNumIter(scf::ForOp forOp, OpBuilder &builder) {
  return getLoopNumIter(forOp.getLowerBound(), forOp.getUpperBound(),
                        forOp.getStep(), forOp.getLoc(), builder);
}

int getBarrierID(ttng::WarpGroupOp wgOp) {
  // get barId atttribute from wgOp
  auto barIdAttr = wgOp->getAttr(ATTR_WS_BARID);
  assert(barIdAttr && "warp group does not have barId attribute");
  auto barId = cast<IntegerAttr>(barIdAttr).getInt();
  return barId;
}

bool isInnerMostLoop(scf::ForOp forOp) {
  return llvm::all_of(forOp.getBody()->without_terminator(),
                      [](Operation &op) { return !isa<scf::ForOp>(op); });
}

bool isMMAOp(Operation *op) {
  return isa<ttng::WarpGroupDotOp, ttng::MMAv5OpInterface>(op);
}

bool isFMHAMathLoop(scf::ForOp forOp) {
  // check if the loop is innermost and have two matmul ops
  if (!isInnerMostLoop(forOp))
    return false;
  int numDotOps = 0;
  int numRedOps = 0;
  forOp->walk([&](Operation *op) {
    if (isMMAOp(op)) {
      ++numDotOps;
    } else if (isa<triton::ReduceOp>(op)) {
      ++numRedOps;
    }
  });
  return ((numDotOps == 2) && (numRedOps >= 2));
}

std::map<std::string, WSGroup> collectGroups(ModuleOp mod) {
  std::map<std::string, WSGroup> groups;
  for (auto namedAttr : mod->getAttrDictionary())
    if (namedAttr.getName().str().find(ATTR_WS_PREFIX) != std::string::npos)
      groups[namedAttr.getName().str()] =
          getGroupFromAttribute(namedAttr.getValue());
  return groups;
}

TokenInfo getTokenProducerOp(Value result) {
  assert(isa<ttg::AsyncTokenType>(result.getType()));
  if (auto blockArg = dyn_cast<BlockArgument>(result)) {
    auto argNum = blockArg.getArgNumber();
    if (auto forOp =
            blockArg.getOwner()->getParent()->getParentOfType<scf::ForOp>()) {
      assert(argNum > 0);
      auto result = forOp.getResults()[argNum - 1];
      return getTokenProducerOp(result);
    } else {
      llvm_unreachable("unhandled op with block-argument");
    }
  }
  auto op = result.getDefiningOp();
  if (auto mmav5 = dyn_cast<ttng::MMAv5OpInterface>(op)) {
    return {mmav5, mmav5.getAccumulator()};
  } else if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(op)) {
    return {tmemAlloc, tmemAlloc.getResult()};
  } else if (auto tmemStore = dyn_cast<ttng::TMEMStoreOp>(op)) {
    return {tmemStore, tmemStore.getDst()};
  } else if (auto tmemLoad = dyn_cast<ttng::TMEMLoadOp>(op)) {
    return {tmemLoad, tmemLoad.getSrc()};
  } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    // if it is for-loop, visit body to find token producer
    // we assume that token is produce in the loop body, and not just carried
    // through
    int resultIdx = -1;
    for (auto [idx, res] : llvm::enumerate(forOp.getResults())) {
      if (res == result) {
        resultIdx = idx;
        break;
      }
    }
    assert(resultIdx >= 0);
    auto token = forOp.getBody()->getTerminator()->getOperand(resultIdx);
    assert(isa<ttg::AsyncTokenType>(token.getType()));
    return getTokenProducerOp(token);
  } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    llvm_unreachable("ifOp is unsupported yet");
  } else if (auto arefPhiOp = dyn_cast<ttng::ArefPhiOp>(op)) {
    return getTokenProducerOp(arefPhiOp.getLocal());
  } else {
    return {};
  }
};

// --------------------------------------------

ttg::MemDescType getDataMemDescType(ttg::MemDescType memDescType,
                                    bool mutableMemory) {
  auto shape = memDescType.getShape();
  SmallVector<int64_t> dataShape(shape.begin() + 1, shape.end());
  return ttg::MemDescType::get(dataShape, memDescType.getElementType(),
                               memDescType.getEncoding(),
                               memDescType.getMemorySpace(), mutableMemory);
};

ttg::MemDescType getArefbufMemDescType(ttg::MemDescType memDescType,
                                       int32_t AREF_SIZE) {
  auto shape = memDescType.getShape();
  SmallVector<int64_t> bufferShape(shape.begin(), shape.end());
  bufferShape.insert(bufferShape.begin(), AREF_SIZE);
  return ttg::MemDescType::get(bufferShape, memDescType.getElementType(),
                               memDescType.getEncoding(),
                               memDescType.getMemorySpace(), true);
}

bool isHopper(ModuleOp mod) {
  auto target = mod->getAttrOfType<StringAttr>(ttg::AttrTargetName);
  return target == "cuda:90";
}

// --------------------------------------------

Value mkConstant(OpBuilder &builder, Location loc, int value, int width,
                 std::set<std::string> groups) {
  auto constValue = builder.create<arith::ConstantIntOp>(loc, value, width);
  if (!groups.empty())
    setGroups(constValue, groups);
  return constValue;
}

bool isConstant(Value value, int constant) {
  if (auto constOp = value.getDefiningOp<arith::ConstantIntOp>())
    if (constOp.value() == constant)
      return true;
  return false;
}

Operation *createAlloc(OpBuilder &builder, Location loc,
                       ttg::MemDescType memDescType, Value src) {
  if (isa<ttg::SharedMemorySpaceAttr>(memDescType.getMemorySpace()))
    return builder.create<ttg::LocalAllocOp>(loc, memDescType, src);
  else {
    assert(isa<ttng::TensorMemorySpaceAttr>(memDescType.getMemorySpace()));
    return builder.create<triton::nvidia_gpu::TMEMAllocOp>(loc, memDescType,
                                                           src);
  }
}

bool isMMAOperandLoadOp(Operation *op) {
  // Check if the given LoadOp can be lowered to cpasync
  if (!isa<LoadOp>(op)) {
    return false;
  }

  LoadOp loadOp = cast<LoadOp>(op);
  auto resType = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
  if (!resType) {
    return false;
  }

  std::function<bool(Operation *)> isUsedByDot = [&](Operation *op) {
    // transitively check if one of the user of the op is a dot op
    if (isa<ttng::MMAv5OpInterface, ttng::WarpGroupDotOp>(op)) {
      return true;
    } else {
      for (auto user : op->getUsers())
        if (isUsedByDot(user)) {
          return true;
        }
      return false;
    }
  };

  // For now, allow cpasync only for loading dot operands
  // When we add support for a specialized warp group for loading additional
  // tensors used in the epilogue, we should relax this condition.
  if (!isOpInGroup(op, ATTR_WS_TMALOAD) && !isUsedByDot(op)) {
    return false;
  }

  auto copyVecBytes = getCopyVecBytes(resType, getSharedEncoding(op));

  // At least 4 bytes needed for cpasync
  return copyVecBytes >= 4;
}

bool isSIMTOp(Operation *op) {
  auto types = llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes());
  for (Type type : types) {
    if (isa<RankedTensorType>(type)) {
      return true;
    }
  }
  return false;
}

} // namespace nvws
} // namespace triton
} // namespace mlir
