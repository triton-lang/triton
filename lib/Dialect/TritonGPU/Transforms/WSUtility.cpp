#include "triton/Dialect/TritonGPU/Transforms/WSUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <optional>

#include "mlir/IR/Builders.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace triton {

namespace ttng = triton::nvidia_gpu;
namespace gpu {

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
  return getGroups(op, ATTR_WSGROUPS);
}

std::set<std::string> getGroups(OpResult result) {
  return getGroups(result.getOwner(),
                   std::string(ATTR_WSGROUPS) + "." +
                       std::to_string(result.getResultNumber()));
}

void setGroups(Operation *op, const std::set<std::string> &groups) {
  setGroups(op, ATTR_WSGROUPS, groups);
}

void setGroups(OpResult result, const std::set<std::string> &groups) {
  setGroups(result.getOwner(),
            std::string(ATTR_WSGROUPS) + "." +
                std::to_string(result.getResultNumber()),
            groups);
}

void addGroups(Operation *op, const std::set<std::string> &groups) {
  addGroups(op, ATTR_WSGROUPS, groups);
}

void addGroups(OpResult result, const std::set<std::string> &groups) {
  auto op = result.getOwner();
  auto attrName = std::string(ATTR_WSGROUPS) + "." +
                  std::to_string(result.getResultNumber());
  addGroups(op, attrName, groups);
}

void copyGroups(Operation *from_op, Operation *to_op) {
  if (from_op->hasAttr(ATTR_WSGROUPS)) {
    setGroups(to_op, ATTR_WSGROUPS, getGroups(from_op, ATTR_WSGROUPS));
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

WSGroup getGroupFromSymbolRefAttr(ModuleOp mod, SymbolRefAttr refAttr) {
  auto attr = mod->getAttr(refAttr.getRootReference());
  assert(attr);
  auto dictAttr = cast<DictionaryAttr>(attr);
  auto startWarp = cast<IntegerAttr>(dictAttr.get("start_warp")).getInt();
  auto numWarps = cast<IntegerAttr>(dictAttr.get("num_warps")).getInt();
  return {(int)startWarp, (int)numWarps};
}

bool isOpInGroup(Operation *op, const std::string &group) {
  if (!op->hasAttr(ATTR_WSGROUPS)) {
    return false;
  }
  auto groups = getGroups(op, ATTR_WSGROUPS);
  return groups.count(group) > 0;
}

bool isResultInGroup(Value value, const std::string &group) {
  auto result = cast<OpResult>(value);
  auto op = result.getOwner();
  auto attrName = std::string(ATTR_WSGROUPS) + "." +
                  std::to_string(result.getResultNumber());
  if (!op->hasAttr(attrName)) {
    return false;
  }
  return getGroups(op, attrName).count(group) > 0;
}

void setGroupAttribute(ModuleOp moduleOp, const std::string &name,
                       int startWarp, int numWarps) {
  assert(!moduleOp->hasAttr(ATTR_WSGROUPS));
  OpBuilder builder(moduleOp.getContext());
  auto attr_pair =
      std::vector<NamedAttribute>{{builder.getStringAttr("start_warp"),
                                   builder.getI32IntegerAttr(startWarp)},
                                  {builder.getStringAttr("num_warps"),
                                   builder.getI32IntegerAttr(numWarps)}};
  NamedAttrList attrList(attr_pair);
  auto dictAttr = DictionaryAttr::get(moduleOp.getContext(), attrList);
  moduleOp->setAttr(name, dictAttr);
}

void setGroupAttribute(ModuleOp moduleOp, const std::string &name,
                       WSGroup group) {
  setGroupAttribute(moduleOp, name, group.startWarp, group.numWarps);
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

} // namespace gpu
} // namespace triton
} // namespace mlir
