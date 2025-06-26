#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/include/TritonAMDGPUTransforms/MfmaGroup.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#undef LLVM_DEBUG
#define LLVM_DEBUG(X) X

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-reschedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
std::string hr(64, '*'); // horizontal rule for debugging

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

/*
 Reschedule ops runs after refine-ops-pass to interleave refined ops at the ttgir level.
 Scheduling consists of multiple passes which:
 (1) Analyse the current op sequence.
 (2) Create an additional set of dependencies in DepMap.
  (a) Deps are stored externally to the Dag, then later added.
 (3) Create a new SchedDag with all prior and new dependencies.
 (4) During scheduling, as ops are scheduled, their nodes are removed from the SchedDag,
     parent/child dependencies are removed from the nodes creating new leaves ready to be scheduled.
 (5) Resulting in a new op sequence.
 

DepMap deps;
SchedDag dag; // create new copy without deps
dag.clearDeps();
now empty nodes without links;
dag.applyDeps(deps);
might want to remove deps.


 */
namespace {

// TODO (ravil): Note, took function from `SchedInstructions.cpp`.
// we need to combine these two implementations
Operation *createSchedBarrier(OpBuilder &rewriter, Location loc,
                              mlir::amdgpu::sched_barrier_opt_enum maskValue) {
  IntegerAttr mask =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(maskValue));
  return rewriter.create<ROCDL::SchedBarrier>(loc, mask);
}

// For cleaner debug printing of scheduling,
// only print the op string which contains operands.
std::string opStringShort(Operation *op) {
  // Get full op string.
  std::string dbgStr;
  llvm::raw_string_ostream dbgStream(dbgStr);
  OpPrintingFlags flags;
  flags.getLargeResourceStringLimit();
  op->print(dbgStream);
  std::string opStrFull = dbgStream.str();
  // If no operands, return full string. E.g. 
  if (op->getNumResults() == 0 && op->getNumOperands() == 0) {
    return opStrFull;
  }
  // Find last index of results and operands.
  int32_t idxLastOperand = 0;
  for (int i = 0; i < op->getNumResults(); i++) {
    // Get string of result.
    Value value = op->getResult(i);
    std::string dbgStr;
    llvm::raw_string_ostream dbgStream(dbgStr);
    value.printAsOperand(dbgStream, flags);
    std::string opdFull = dbgStream.str();
    int32_t opdEndIdx = opdFull.find(" ");
    std::string opdName = opdFull.substr(0, opdEndIdx);
    // Find result in original string.
    int32_t idx = opStrFull.find(opdName) + opdName.size();
    if (idx > idxLastOperand)
      idxLastOperand = idx;
  }
  for (int i = 0; i < op->getNumOperands(); i++) {
    // Get string of operand.
    Value value = op->getOperand(i);
    std::string dbgStr;
    llvm::raw_string_ostream dbgStream(dbgStr);
    value.printAsOperand(dbgStream, flags);
    std::string opdFull = dbgStream.str();
    int32_t opdEndIdx = opdFull.find(" ");
    std::string opdName = opdFull.substr(0, opdEndIdx);
    // Find operand in original string.
    int32_t idx = opStrFull.find(opdName) + opdName.size();
    if (idx > idxLastOperand)
      idxLastOperand = idx;
  }
  return opStrFull.substr(0, idxLastOperand);
}

enum class SchedulingDirection { TopDown, BottomUp };


/******************************************************************************
 Data dependency points in the opposite direction of data flow.

 A child node [data] depends on parent node because data flows from parent to child.


 dst / parent / dependee
  ^
  |
 src / child / dependent

 */

/*
 Node contains op, parents and children dependencies.
 Before scheduling Nodes are created, dependencies are added.
 During scheduling, each node scheduled removes a node from the dag,
 and that node's children get it removed as a parent.
 Nodes without parents are ready to be scheduled.
******************************************************************************/
struct SchedDagNode {
  SchedDagNode(Operation *op) : op(op) {
    static int32_t serialId = 0;
    id = serialId++;
    opStr = opStringShort(op);
  }

  void addChild(SchedDagNode *node) {children.insert(node); }
  void addParent(SchedDagNode *node) { parents.insert(node); }
  Operation *getOp() { return op; }
  bool hasChildren() { return !children.empty(); }
  bool hasParents() { return !parents.empty(); }
  int32_t numChildren() { return children.size(); }
  int32_t numParents() { return parents.size(); }

  /*
  When scheduling top-down, a node is ready to schedule when it has no parents.
  When scheduling bottom-up, a node is ready to schedule when it has no children.
  */
  template <SchedulingDirection Direction>
  bool isReady() {
    if constexpr (Direction == SchedulingDirection::TopDown) {
     return parents.empty();
    } else {
      return children.empty();
    }
  }

  const llvm::SetVector<SchedDagNode *> &getChildren() { return children; }
  const llvm::SetVector<SchedDagNode *> &getParents() { return parents; }


  bool removeChild(SchedDagNode *node) {
    if (children.contains(node)) {
      children.remove(node);
      return true;
    }
    return false;
  }

  bool removeParent(SchedDagNode *node) {
    if (parents.contains(node)) {
      parents.remove(node);
      return true;
    }
    return false;
  }

  void clearDeps() {
    children.clear();
    parents.clear();
  }

  Operation *op;
  int32_t id; // for debug printing
  std::string opStr;
  /*
  Children depend on this node; this node must be scheduled before children.
  */
  llvm::SetVector<SchedDagNode *> children;

  // This node depends on parents; this node must be scheduled after parents.
  llvm::SetVector<SchedDagNode *> parents;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &out, SchedDagNode &node) {
  out << "[@" << node.id << " " << node.opStr;
  out << " p={";
  for (auto p : node.getParents()) {
    out << "@" << p->id << " ";
  }
  out << "} c={";
  for (auto c : node.getChildren()) {
    out << "@" << c->id << " ";
  }
  out << "}]";
  return out;
}
typedef SmallVector<SchedDagNode *> SchedDagNodeList;

/******************************************************************************
  Creates list of nodes using shared_ptr to keep a single copy.
  Also creates a list of plain node pointers for algorithm.
******************************************************************************/
struct BasicBlockNodeMap {
  BasicBlockNodeMap(Block *mlirBlock) {
    LDBG("Node to Op mapping for BB");
    for (auto it = mlirBlock->begin(); it != mlirBlock->end(); ++it) {
      Operation *op = &(*it);
      // same as calling new
      std::shared_ptr<SchedDagNode> node = std::make_shared<SchedDagNode>(op);

      //std::string dbgStr;
      //llvm::raw_string_ostream dbgStream(dbgStr);
      //op->print(dbgStream);
      //LDBG(*node << " -> " << dbgStream.str());

      lookup.insert({op, node.get()});
      nodes.push_back(std::move(node));
    }
  }

  // Returns a copy of original node list.
  SchedDagNodeList getNodeList() const {
    SchedDagNodeList nodeList;
    for (auto node : nodes) {
      nodeList.push_back(node.get());
    }
    return nodeList;
  }

  // Lookup node by op.
  SchedDagNode * operator[]( Operation * op) const {
    if (!lookup.contains(op))
      return nullptr;
    return lookup.find(op)->second;
  }

  llvm::SmallVector<std::shared_ptr<SchedDagNode>> nodes;
  llvm::MapVector<Operation *, SchedDagNode *> lookup;
};


// Dependency is from src/child to dst/parent.
// dst must preceed src during scheduling.
// dst / parent
//  ^
//  |
// src / child
struct SchedDep {
  SchedDagNode *parent; // dst
  SchedDagNode *child; // src
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &out, SchedDep &dep) {
  return  out << *(dep.child) << " -> " << *(dep.parent);
}

/* Dependencies are stored in a map to track what they represent.
It also allows for different sets
DenseMap{
 "RefinedOpOrder": [[0,1], [2,3], ...];
 "GlobalLoadLatency": [[4,5], [6,7], ...];
 "LocalStoreLatency": [[8,9], [10,11], ...];
}
 */
typedef SmallVector<SchedDep> DepList;
typedef DenseMap<StringRef, DepList> DepMap;

/*
  Abstract base class.
  Input is serial list of nodes.
  Output is a list of dependencies.
  Child classes must override the calcDeps() class.
*/
struct DependencyCalculator {
  DependencyCalculator(StringRef name, SchedDagNodeList nodes) : depName(name), nodeList(nodes) {}
  virtual void calcDeps() = 0;
  DepList getDepList() {
    return depList;
  }
  StringRef depName;
  SchedDagNodeList nodeList;
  DepList depList;
};


/******************************************************************************
* Create data dependencies based on def-use chains.
* Also creates dependencies based on various barriers.
* This class represents the minimum set of dependencies needed for correctness.
******************************************************************************/
struct DataDependencyCalculator : DependencyCalculator {
  DataDependencyCalculator(SchedDagNodeList nodeList, BasicBlockNodeMap nodeMap) :
      DependencyCalculator("Data", nodeList),
      nodeMap(nodeMap) {
  }

  // There is an implied data dependency between LDS ops and GPUBarrier.
  template <SchedulingDirection Direction> void calcDepsLdsGpuBar() {

    auto fwIt = nodeList.begin();
    auto bkIt = nodeList.rbegin();
    auto next = [&]() -> SchedDagNode * {
      if constexpr (Direction == SchedulingDirection::TopDown) {
        if (fwIt == nodeList.end())
          return nullptr;
        return *(fwIt++);
      }
      if constexpr (Direction == SchedulingDirection::BottomUp) {
        if (bkIt == nodeList.rend())
          return nullptr;
        return *(bkIt++);
      }
      return nullptr;
    };

    llvm::SmallVector<SchedDagNode *> ldsOpsNodes;
    while (SchedDagNode *node = next()) {
      auto localLoad = dyn_cast<triton::gpu::LocalLoadOp>(node->getOp());
      auto localStore = dyn_cast<triton::gpu::LocalStoreOp>(node->getOp());
      auto localAlloc = dyn_cast<triton::gpu::LocalAllocOp>(node->getOp());
      if (localLoad || localStore || localAlloc) {
        ldsOpsNodes.push_back(node);
      }
      auto gpuBarrier = dyn_cast<mlir::gpu::BarrierOp>(node->getOp());
      if (gpuBarrier) {
        SchedDagNode *barrierNode = node;
        for (auto ldsOpNode : ldsOpsNodes) {
          if constexpr (Direction == SchedulingDirection::TopDown) {
            SchedDep dep;
            dep.parent = ldsOpNode;
            dep.child = barrierNode;
            depList.push_back(dep);
          }
          if constexpr (Direction == SchedulingDirection::BottomUp) {
            SchedDep dep;
            dep.parent = barrierNode;
            dep.child = ldsOpNode;
            depList.push_back(dep);
          }
        }
        ldsOpsNodes.clear();
      }
    }
  }

  // GpuBar can't be recordered across themselves.
  // While this may be logically superfluous, it's fine to leave it for clarity.
  void calcDepsGpuBarGpuBar() {
    SchedDagNode *prevBar = nullptr;
    for (auto it = std::next(nodeList.begin()); it != nodeList.end(); ++it) {
      auto node = *it;
      auto bar = dyn_cast<mlir::gpu::BarrierOp>(node->getOp());
      if (bar) {
        if (prevBar) {
          SchedDep dep;
          dep.parent = prevBar;
          dep.child = node;
          depList.push_back(dep);
        }
        prevBar = node;
      }
    }
  }

  // Add data deps for operands.
  void calcDepsOperands() {
    for (auto it = nodeList.begin(); it != nodeList.end(); ++it) {
      SchedDagNode *node = (*it);
      for (auto operandValue : node->getOp()->getOperands()) {
        auto operandDefOp = operandValue.getDefiningOp();
        SchedDagNode *parentNode = nodeMap[operandDefOp];
        if (parentNode) {
          SchedDep dep;
          dep.parent = parentNode;
          dep.child = node;
          depList.push_back(dep);
        }
      }
    }
  }

  // Nodes without results still must come before cf.br.
  void calcDepsCfBr() {
    SchedDagNode *lastNode = (*(nodeList.rbegin())); // cf.br
    for (auto it = std::next(nodeList.rbegin()); it != nodeList.rend(); ++it) {
      SchedDagNode *node = (*it);
      if (node->getOp()->getNumResults() == 0) {
        SchedDep dep;
        dep.parent = node;
        dep.child = lastNode;
        depList.push_back(dep);
      }
    }
  }

  // Sched.bars block ops based on type.
  void calcDepsSchedBar() {
    // TODO(dtanner)
  }

  // SetPrio is like sched.bar(0), no instructions can move past.
  void calcDepsSetPrio() {
    // TODO(dtanner)
  }

  void calcDeps() {
    LDBG("DataDependencyCalculator::calcDeps()");
    calcDepsOperands();
    calcDepsLdsGpuBar<SchedulingDirection::BottomUp>();
    calcDepsLdsGpuBar<SchedulingDirection::TopDown>();
    calcDepsGpuBarGpuBar();
    calcDepsCfBr();
    calcDepsSetPrio();
  }

  BasicBlockNodeMap nodeMap;
};

/******************************************************************************
* Create order dependencies between ops which were refined from same original op.
******************************************************************************/
struct RefinedOpDependencyCalculator : DependencyCalculator {
  RefinedOpDependencyCalculator(SchedDagNodeList nodeList, BasicBlockNodeMap nodeMap) :
      DependencyCalculator("Order", nodeList),
      nodeMap(nodeMap) {
  }

  /*
  * Add dependencies between all dots; it is assumed that they were originally
  * created in the ideal order during refinement.
  * If it changes that they aren't created in ideal order, then we would need to enhance this
  * to place dependencies according to dot tiling.
  * Add deps in order of tileSerial then elementSerial for same dotId.
  */
  void calcDepsDot() {
    SchedDagNode *prevDot = nullptr;
    int32_t prevTileSerial = -1;
    int32_t prevElemSerial = -1;
    for (auto it = nodeList.begin(); it != nodeList.end(); ++it) {
      SchedDagNode *node = *it;
      if (DotOp op = dyn_cast<DotOp>(node->getOp())) {
        if (op->hasAttr(triton::amdgpu::DotTileAttr::getMnemonic())) {
          auto attr = op->getAttrOfType<triton::amdgpu::DotTileAttr>(triton::amdgpu::DotTileAttr::getMnemonic());
          int32_t tileSerial = attr.getTileSerial();
          int32_t elemSerial = attr.getElementSerial();
          LDBG("Found DotOp w/ DotTileAttr " << tileSerial << ", " << elemSerial);
          // When both tile and element serial ids decrease, it means we've started a new dot.
          if (prevDot && (tileSerial >= prevTileSerial || elemSerial >= prevElemSerial)) {
            SchedDep dep;
            dep.parent = prevDot;
            dep.child = node;
            depList.push_back(dep);
          }
          prevDot = node;
          prevTileSerial = tileSerial;
          prevElemSerial = elemSerial;
        } else {
          LDBG("WARNING DotOp has no DotTileAttr:" << *node);
        }
      }
    }
  }

  /*
  * Add dependencies between local loads which
  * (1) Belong to the same dot.
  * (2) Belong to the same operand of the dot.
  * It is assumed that the above two criterial mean the local loads would have been refined
  * from the same unrefined local load.
  * It is also assumed that the ideal relative order of refined local loads
  * is monotonically increasing addresses.
  */
  void calcDepsLocalLoad() {
    SchedDagNode *prevNode = nullptr;
    int32_t prevTileSerial = -1;
    int32_t prevElemSerial = -1;
    int32_t prevOpdIdx = -1;
    for (auto it = nodeList.begin(); it != nodeList.end(); ++it) {
      SchedDagNode *node = *it;
      if (ttg::LocalLoadOp op = dyn_cast<ttg::LocalLoadOp>(node->getOp())) {
        auto resultType = cast<RankedTensorType>(op.getType());
        auto resultEncode = cast<DotOperandEncodingAttr>(resultType.getEncoding());
        auto opdIdx = resultEncode.getOpIdx();
        LDBG("Found LocalLoadOp w/ opdIdx=" << opdIdx << "; " << *node);
        if (op->hasAttr(triton::amdgpu::DotTileAttr::getMnemonic())) {
          auto attr = op->getAttrOfType<triton::amdgpu::DotTileAttr>(triton::amdgpu::DotTileAttr::getMnemonic());
          int32_t tileSerial = attr.getTileSerial();
          int32_t elemSerial = attr.getElementSerial();
          LDBG("Found LocalLoadOp w/ DotTileAttr " << tileSerial << ", " << elemSerial);

          if (prevNode && (tileSerial >= prevTileSerial || elemSerial >= prevElemSerial) && opdIdx == prevOpdIdx) {
            SchedDep dep;
            dep.parent = prevNode;
            dep.child = node;
            depList.push_back(dep);
          }
          prevNode = node;
          prevTileSerial = tileSerial;
          prevElemSerial = elemSerial;
          prevOpdIdx = opdIdx;
        } else {
          LDBG("WARNING LocalLoadOp has no DotTileAttr:" << *node);
        }
      }
    }
  }

  /*
  * Add dependencies between ops which were refined from the same op.
  * It is assumed that the ideal relative order of refines ops is the order created during refinement.
  */
  void calcDepsRefinedOp() {
    SchedDagNode *prevNode = nullptr;
    int32_t prevId = -1;
    for (auto it = std::next(nodeList.begin()); it != nodeList.end(); ++it) {
      SchedDagNode *node = *it;
      Operation *op = node->getOp();
      if (op->hasAttr(triton::amdgpu::RefinedOpOrderAttr::getMnemonic())) {
        auto attr = op->getAttrOfType<triton::amdgpu::RefinedOpOrderAttr>(triton::amdgpu::RefinedOpOrderAttr::getMnemonic());
        int32_t idOriginalOp = attr.getIdOriginalOp();
        int32_t idRefinedOp = attr.getIdRefinedOp();
        LDBG("Found RefinedOp: " << idOriginalOp << ", " << idRefinedOp);
        if (prevNode && idOriginalOp == prevId) {
          SchedDep dep;
          dep.parent = prevNode;
          dep.child = node;
          depList.push_back(dep);
        }
        prevNode = node;
        prevId = idOriginalOp;
      }
    }
  }

  void calcDeps() {
    calcDepsDot();
    calcDepsLocalLoad(); // TODO(dtanner) add me and verify Ravil's canonicalizer passing forward attributes.
    calcDepsRefinedOp();
  }

  BasicBlockNodeMap nodeMap;
};

/******************************************************************************
  SchedDag consists of nodes containing edges to other nodes.
  DepMap stored outside of Dag.
******************************************************************************/
struct SchedDag {
public:
  SchedDag(SchedDagNodeList nodes) : nodes(nodes) {
    // When creating a new dag from nodes, clear dependencies from nodes.
    resetDeps();
  }

  SmallVector<SchedDagNode *> getNodes() {
    SmallVector<SchedDagNode *> copy(nodes.size(), nullptr);
    for (auto [idx, node] : llvm::enumerate(nodes)) {
      copy[idx] = node;
    }
    return copy;
  }


  void addAllDeps(const DepMap & deps) {
    // Reset deps
    resetDeps();
    LDBG("SchedDag::addAllDeps() child -> parent.");
    for (auto& depType : deps) {
      StringRef depTypeName = depType.getFirst();
      LDBG("DepTypeName: " << depTypeName);
      DepList depList = depType.getSecond();
      for (auto dep : depList) {
        LDBG("    " << dep);
        dep.child->addParent(dep.parent);
        dep.parent->addChild(dep.child);
      }
    }
  }

  void resetDeps() {
    for (auto [idx, node] : llvm::enumerate(nodes)) {
      node->clearDeps();
    }
  }

  template <SchedulingDirection Direction>
  void initReadyNodes() {
    readyNodes.clear();
    for (auto node : nodes) {
      if (node->isReady<Direction>()) {
        readyNodes.insert(node);
      }
    }
  }

  bool finished() { return readyNodes.empty(); }

  /*
  * After scheduling a node top-down, mark it's children as dependency-fulfilled.
  * After scheduling a node bottom-up, mark it's parents as dependency-fulfilled.
  */
  template <SchedulingDirection Direction>
  void removeScheduledNode(SchedDagNode *node) {
    assert(node->isReady<Direction>());
    readyNodes.remove(node);

    if constexpr (Direction == SchedulingDirection::TopDown) {
      for (auto child : node->getChildren()) {
        child->removeParent(node);
        if (child->isReady<Direction>()) {
          readyNodes.insert(child);
        }
      }
    } else {
      for (auto parent : node->getParents()) {
        parent->removeChild(node);
        if (parent->isReady<Direction>()) {
          readyNodes.insert(parent);
        }
      }
    }
  }

  SetVector<SchedDagNode *> &getReadyNodes() {
    return readyNodes;
  }

private:

  SchedDagNodeList nodes;
  SetVector<SchedDagNode *> readyNodes;

};

llvm::raw_ostream &operator<<(llvm::raw_ostream &out, SchedDag &dag) {
  out << "digraph \"dep-dag\" {\n";
  out << "rankdir=\"LR\"\n";
  for (auto [idx, node] : llvm::enumerate(dag.getNodes())) {
    std::string name = std::to_string(reinterpret_cast<intptr_t>(node));
    out << name << "\t[label=\"" << node->getOp()->getName() << "\"]\n";
  }
  for (auto [idx, node] : llvm::enumerate(dag.getNodes())) {
    std::string name = std::to_string(reinterpret_cast<intptr_t>(node));
    for (auto child : node->getChildren()) {
      std::string childName = std::to_string(reinterpret_cast<intptr_t>(child));
      out << "\t" << childName << " -> " << name << ";\n";
    }
  }
  out << "}";
  return out;
}

/******************************************************************************
******************************************************************************/
struct SchedManager {
  SchedManager(Block *block) :
      nodeMap(block),
      nodeList(nodeMap.getNodeList()),
      dag(nodeList) {}

  void addDataDeps() {
    LDBG("SchedManager::addDataDeps()");
    DataDependencyCalculator dataDeps(nodeList, nodeMap);
    dataDeps.calcDeps();
    deps[dataDeps.depName] = dataDeps.getDepList();
  }

  void addRefinedDeps() {
    LDBG("SchedManager::addRefinedDeps()");
    RefinedOpDependencyCalculator refinedDeps(nodeList, nodeMap);
    refinedDeps.calcDeps();
    deps[refinedDeps.depName] = refinedDeps.getDepList();
  }

  void printNodes() {
    LDBG("nodeList:");
    for (auto it = nodeList.begin(); it != nodeList.end(); ++it) {
      auto node = *it;
      LDBG(*node);
    }
  }

  template <SchedulingDirection Direction>
  void reschedule() {
    static int32_t rescheduleId = 0;
    LDBG("");
    LDBG(hr);
    LDBG("SchedManager::reschedule(" << rescheduleId << ") " << ((Direction == SchedulingDirection::TopDown) ? "TopDown" : "BottomUp"));
    LDBG(hr);

    LDBG("");
    LDBG(hr);
    LDBG("Adding deps to dag before reschedule(" << rescheduleId << ")");
    dag.addAllDeps(deps);
    // LDBG("Dependency dag in dot-format:\n" << dag);


    LDBG("");
    LDBG(hr);
    LDBG("NodeList before reschedule(" << rescheduleId << ")");
    LLVM_DEBUG(printNodes());

    // Node readiness is based on direction.
    dag.initReadyNodes<Direction>();

    // Schedule the dag.
    LDBG("");
    LDBG(hr);
    // Store nodes is newly scheduled order.
    SchedDagNodeList rescheduledNodes;
    const bool printReadyList = true;
    for (int iter = 0; !dag.finished(); ++iter) {
      // if (iter > 30) break; // TODO(dtanner) remove me
      const auto &readyNodes = dag.getReadyNodes();
      SchedDagNode *selectedNode = selectFromReadyNodes<Direction>(readyNodes);

      LDBG("reschedule(" << rescheduleId << ") Iter:" << iter << "; Ready: " << readyNodes.size() << "; Selected: " << *selectedNode);
      if (printReadyList) {
        std::string dbgStr;
        llvm::raw_string_ostream dbgStream(dbgStr);
        dbgStream << "Ready List:\n";
        for (auto node : readyNodes) {
          dbgStream << *node << "\n";
        }
        LDBG(dbgStream.str());
      }
      // Place selected node in list, remove it from dag which updates readyList.
      rescheduledNodes.push_back(selectedNode);
      dag.removeScheduledNode<Direction>(selectedNode);
    }

    // Update nodeList after rescheduling.
    if constexpr (Direction == SchedulingDirection::TopDown) {
      nodeList = rescheduledNodes;
    } else {
      nodeList.clear();
      for (auto it = rescheduledNodes.rbegin(); it != rescheduledNodes.rend(); ++it) {
        auto &node = *it;
        nodeList.push_back(node);
      }
    }
    LDBG("");
    LDBG(hr);
    LDBG("NodeList after reschedule(" << rescheduleId << ")");
    LLVM_DEBUG(printNodes());
    LDBG(hr);
    LDBG("SchedManager::reschedule(" << rescheduleId << ") - DONE");
    LDBG(hr);
    rescheduleId++;
  }

  /******************************************************************************
  * Comparators for scheduling from ready list.
  * Scheduling Early/Late refers to closer to the top or bottom of the instruction order
  * and will depend on SchedulingDirection.
  * ASAP refers to getting the op out of the ready list as soon as possible;
  * so top-down this means early, and bottom-up this means late.
  * We do want to get to dots as soon as possible.
  * Have the scheduler naturally minimize regalloc by
  * (1) Loads as late as possible.
  * (2) Stores as early as possible.
  ******************************************************************************/

  // Op lowers to nop, or nearly nop;
  // schedule these asap to put "real" ops into ready list asap.
  bool opTypeIsNoOp(SchedDagNode *node) {
    Operation *op = node->getOp();
    return llvm::isa<
        triton::gpu::MemDescSubviewOp,
        triton::gpu::MemDescTransOp,
        tt::amdgpu::ExtractSliceOp,
        tt::amdgpu::ConcatOp,
        mlir::gpu::BarrierOp
      >(op);
  }

  // Schedule nops asap (meaning closer to beginning or end of bb).
  bool opTypeIsLoad(SchedDagNode *node) {
    Operation *op = node->getOp();
    return llvm::isa<
        triton::LoadOp,
        triton::gpu::LocalLoadOp,
        triton::amdgpu::BufferLoadOp
      >(op);
  }

  // Schedule nops asap (meaning closer to beginning or end of bb).
  bool opTypeIsStore(SchedDagNode *node) {
    Operation *op = node->getOp();
    return llvm::isa<
    triton::LoadOp,
    triton::gpu::LocalStoreOp
      >(op);
  }

  // Returns true if a is load and b isn't (for bottom-up).
  template<SchedulingDirection Direction>
  bool readyNodeCompareLoad(SchedDagNode *a, SchedDagNode *b) {
    return (opTypeIsLoad(a) and !opTypeIsLoad(b)) == (Direction == SchedulingDirection::BottomUp);
  }
  
  // Returns true if a is store and b isn't (for top-down).
  template<SchedulingDirection Direction>
  bool readyNodeCompareStore(SchedDagNode *a, SchedDagNode *b) {
    return (opTypeIsStore(a) and !opTypeIsStore(b)) == (Direction == SchedulingDirection::TopDown);
  }

  // Returns true if a is nop and b isn't.
  template<SchedulingDirection Direction>
  bool readyNodeCompareNop(SchedDagNode *a, SchedDagNode *b) {
    return (opTypeIsNoOp(a) and !opTypeIsNoOp(b))== (Direction == SchedulingDirection::BottomUp);
  }

  // Returns true if a will add more new nodes to ready list than b.
  template<SchedulingDirection Direction>
  bool readyNodeCompareReadyListSize(SchedDagNode *a, SchedDagNode *b) {
    if (Direction == SchedulingDirection::TopDown) {
      return a->numChildren() > b->numChildren();
    } else {
      return a->numParents() > b->numParents();
    }
  }

  // Returns true if a before b in original mlir block order (for top down).
  template<SchedulingDirection Direction>
  bool readyNodeCompareOriginalOrder(SchedDagNode *a, SchedDagNode *b) {
    return (a->id < b->id) == (Direction == SchedulingDirection::TopDown);
  }

  /*
  Returns which of the two ops is preferred to schedule now.
  Compares based on boolean comparison functions 
  */
  template<SchedulingDirection Direction>
  SchedDagNode *readyNodeCompare(SchedDagNode *a, SchedDagNode *b) {

    // Compare based on nops; prefer to schedule nops asap
    // so that real ops enter the ready nodes.
    if (readyNodeCompareNop<Direction>(a, b)) {
      return a;
    } else if (readyNodeCompareNop<Direction>(b, a)) {
      return b;
    }

    // Compare based on loads as late as possible.
    if (readyNodeCompareLoad<Direction>(a, b)) {
      return a;
    } else if (readyNodeCompareLoad<Direction>(b, a)) {
      return b;
    }

    // Compare based on stores as early as possible.
    if (readyNodeCompareStore<Direction>(a, b)) {
      return a;
    } else if (readyNodeCompareStore<Direction>(b, a)) {
      return b;
    }

    // Compare based on how many new nodes will be placed into ready lst.
    if (readyNodeCompareReadyListSize<Direction>(a, b)) {
      return a;
    } else if (readyNodeCompareReadyListSize<Direction>(b, a)) {
      return b;
    }

    // Final comparison based on original mlirBlock order.
    if (readyNodeCompareOriginalOrder<Direction>(a, b)) {
      return a;
    }
    return b;
  }

  /*
  Select best node from ready list.
  */
  template<SchedulingDirection Direction>
  SchedDagNode *selectFromReadyNodes(SetVector<SchedDagNode *> readyNodes) {
#if 1
    // Select random node to validate data dependencies.
    int32_t idx = rand()%readyNodes.size();
    return readyNodes[idx];
#endif
    SchedDagNode *selected = readyNodes.front();
    for (auto it = std::next(readyNodes.begin()); it != readyNodes.end(); ++it) {
      SchedDagNode *node = *it;
      selected = readyNodeCompare<Direction>(selected, node);
    }
    return selected;
  }

  SmallVector<Operation *> getOpList() {
    SmallVector<Operation *> opList;
    for (auto node : nodeList) {
      Operation *op = node->getOp();
      opList.push_back(op);
    }
    return opList;
  }

private:
  BasicBlockNodeMap nodeMap; // does not get changed; contains shared_ptrs.
  SchedDagNodeList nodeList; // rewritten every rescheduling.
  SchedDag dag;
  DepMap deps;
  DenseMap<SchedDagNode *, size_t> nodeIndices;

}; // SchedManager

struct TritonAMDGPURescheduleOps
    : public TritonAMDGPURescheduleOpsBase<TritonAMDGPURescheduleOps> {
  explicit TritonAMDGPURescheduleOps(StringRef targetArch) {
    this->arch = targetArch.str();
  }

  LogicalResult verify(Block *mlirBlock) {
    // make sure that a block gets terminated with `cf::BranchOp`
    if (!dyn_cast<mlir::cf::BranchOp>(&(mlirBlock->back()))) {
      return failure();
    }
    
    // don't schedule if there is not enough operations in a block
    if (mlirBlock->getOperations().size() < 3)
      return failure();
    return success();
  }

  /*
    applyReschedulingPasses() is the top-level scheduling pass for a single block,
    whose purpose is to improve performance and regalloc of backend compilers.
    Before this pass, mfmas and local_loads (belonging to dots)
    were already annotated with their dot-tile info.
    The order of re-scheduling is:
     - Place order dependencies on dots according to dot-tiling.
     - Place order dependencies on local_loads according to dot-tiling.
     - Determine min-register vs max-latency-hiding preference.
     - Determine memory op order and co-scheduling.
     - Place order dependencies between memory ops.
     - Determine memory ops' early/late preference.
     - Determine memory ops' preferred issue rate.
     - Determine memory ops' supported issue rate.
     - Place performance and anti-dependencies between memory ops and dots.
     - Run scheduler with new dependencies in place.
    Note that reschedule() can be run after any new dependencies are created to
    visualize dag.
  */
  void applyReschedulingPasses(Block *mlirBlock) {
    LDBG("");
    LDBG("");
    LDBG("TritonAMDGPURescheduleOps::applyReschedulingPasses()");

    SchedManager schedManager(mlirBlock);
    schedManager.addDataDeps();
    schedManager.addRefinedDeps();
    schedManager.reschedule<SchedulingDirection::BottomUp>();

    SmallVector<Operation *> rescheduledOps = schedManager.getOpList();

    std::string outStr;
    llvm::raw_string_ostream outStream(outStr);
    for (auto op : rescheduledOps) {
      op->print(outStream);
      outStream << "\n";
    }
    LDBG("");
    LDBG(hr);
    LDBG("Rescheduled Ops:");
    LDBG(outStream.str());

    // Re-order instruction based on the new schedule;
    // preserving order but going from last to first op.
    // Move instruction from the tail of the rescheduled list
    // to the begining of the current BB.
    for (auto it = rescheduledOps.rbegin(); it != rescheduledOps.rend(); ++it) {
      (*it)->moveBefore(mlirBlock, mlirBlock->begin());
    }
    
    for (auto &op : mlirBlock->getOperations()) {
      LLVM_DEBUG(op.dump());
    }
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    llvm::SmallVector<Block *> blocks;
    mod.walk([&](triton::amdgpu::InstructionSchedHint hint) {
      if (hint.getVariant() == triton::amdgpu::SchedHint::refine_ops) {
        blocks.push_back(hint->getBlock());
        hint->erase();
      }
    });

    for (auto block : blocks) {
      if (succeeded(verify(block))) {
        applyReschedulingPasses(block);
      }
    }
  }
};
} // namespace

namespace mlir {
std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPURescheduleOpsPass(StringRef targetArch) {
  return std::make_unique<TritonAMDGPURescheduleOps>(targetArch);
}
} // namespace mlir
