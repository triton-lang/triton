#include "WSUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <optional>

#include "mlir/IR/Builders.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace triton {

namespace ttng = triton::nvidia_gpu;
namespace gpu {


SmallVector<ttng::WarpGroupOp> findWarpGroupOps(ModuleOp m) {
  SmallVector<ttng::WarpGroupOp> wgOps;
  m.walk([&](ttng::WarpGroupOp wgOp) {
      wgOps.push_back(wgOp);
  });
  return wgOps;
}

std::vector<SymbolRefAttr> getGroupsAttr(mlir::Operation *op,
                                         std::string attrName) {
  std::vector<SymbolRefAttr> attrs;
  auto attr = op->getAttr(attrName);
  for (auto attr : mlir::cast<mlir::ArrayAttr>(op->getAttr(attrName))) {
    attrs.push_back(mlir::cast<SymbolRefAttr>(attr));
  }
  return attrs;
}

std::set<std::string> getGroups(mlir::Operation *op, std::string attrName) {
  std::set<std::string> groups;
  for (auto attr : getGroupsAttr(op, attrName)) {
    groups.insert(attr.getRootReference().str());
  }
  return groups;
}

void setGroups(mlir::Operation *op, std::string attrName,
               std::set<std::string> groups) {
  mlir::SmallVector<mlir::Attribute, 4> attrs;
  auto mod = op->getParentOfType<ModuleOp>();
  for (auto group : groups) {
    attrs.push_back(mlir::SymbolRefAttr::get(op->getContext(), group));
  }
  op->setAttr(attrName, mlir::ArrayAttr::get(op->getContext(), attrs));
}

WSGroup getGroupFromSymbolRefAttr(ModuleOp mod, SymbolRefAttr refAttr) {
  mlir::Attribute attr = mod->getAttr(refAttr.getRootReference());
  assert(attr);
  auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(attr);
  assert(dictAttr);
  auto startWarp = mlir::cast<IntegerAttr>(dictAttr.get("start_warp")).getInt();
  auto numWarps = mlir::cast<IntegerAttr>(dictAttr.get("num_warps")).getInt();
  return WSGroup{int(startWarp), int(numWarps)};
}

bool isOpInGroup(mlir::Operation *op, std::string group) {
  if (!op->hasAttr(ATTR_WSGROUPS)) {
    return false;
  }
  auto groups = getGroups(op, ATTR_WSGROUPS);
  return groups.count(group) > 0;
}

void setGroupAttribute(ModuleOp moduleOp, std::string name, WSGroup group) {
  assert(!moduleOp->hasAttr(ATTR_WSGROUPS));
  OpBuilder builder(moduleOp.getContext());
  auto attr_pair = std::vector<mlir::NamedAttribute>{
      {builder.getStringAttr("start_warp"),
       builder.getI32IntegerAttr(group.startWarp)},
      {builder.getStringAttr("num_warps"),
       builder.getI32IntegerAttr(group.numWarps)}};
  NamedAttrList attrList(attr_pair);
  auto dictAttr = mlir::DictionaryAttr::get(moduleOp.getContext(), attrList);
  moduleOp->setAttr(name, dictAttr);
}

SymbolRefAttr mkGroup(ModuleOp moduleOp, std::string name, WSGroup group) {
  if (!moduleOp->hasAttr(name)) {
    // if module has group set, just return reference
    setGroupAttribute(moduleOp, name, group);
  }
  auto context = moduleOp.getContext();
  auto attrRef = mlir::SymbolRefAttr::get(context, name);
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
  auto barId = mlir::cast<mlir::IntegerAttr>(barIdAttr).getInt();
  return barId;
}

bool isInnerMostLoop(scf::ForOp forOp) {
  return llvm::all_of(forOp.getBody()->without_terminator(),
                      [](Operation &op) { return !isa<scf::ForOp>(op); });
}

bool isMMAOp(Operation *op) {
  return isa<mlir::triton::nvidia_gpu::WarpGroupDotOp,
             mlir::triton::nvidia_gpu::MMAv5OpInterface>(op);
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
