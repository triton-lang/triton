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

/******************************************************************************
  Reschedule ttgir after refine-ops-pass to interleave refined ops at the ttgir level.

  The goals of rescheduling the basic block are:
  (1) Triton scheduling only needs to compliment LLVM scheduler,
      not redo same heuristics.
  (2) To create a better op order before LLIR to help LLVM backend scheduler.
  (3) Inject sched.barriers into op order to provide scheduling guard rails
      to backend scheduler to constrain pre-RA and post-RA scheduling.

  Scheduling consists of multiple passes which:
  (1) Analyse the current op sequence.
  (2) Create an additional set of dependencies in DepMap.
    (a) Deps are stored externally to the Dag, then later added.
  (3) Populate SchedDag with all prior and new dependencies.
  (4) During readyList-based scheduling, as ops are scheduled,
      their nodes are removed from the SchedDag,
      parent/child dependencies are removed from the nodes
      creating new nodes ready to be scheduled.
  (5) Resulting in a new op sequence.
******************************************************************************/
namespace {

// TODO (ravil): Note, took function from `SchedInstructions.cpp`.
// we need to combine these two implementations
Operation *createSchedBarrier(OpBuilder &rewriter, Location loc,
                              mlir::amdgpu::sched_barrier_opt_enum maskValue) {
  IntegerAttr mask =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(maskValue));
  return rewriter.create<ROCDL::SchedBarrier>(loc, mask);
}

// For abbreviated debug printing during scheduling,
// only print the portion of the op string which contains operands.
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

enum class SchedDirection { TopDown, BottomUp };
enum class SchedPriority { MinRegs, MaxLatencyHiding };

/******************************************************************************
 Dependency points in the opposite direction of data flow.

 The child node depends on parent node because data flows from parent to child;
 dependency "arrow" points from child to parent.

 dst / parent / dependee
  ^
  |
 src / child / dependent
******************************************************************************/

/******************************************************************************
  SchedDagNode contains op, parents and children dependencies.
  Before scheduling nodes are created and dependencies are added.
  During scheduling, each node scheduled removes it from the dag,
  and that node's deps are removed.
  Nodes without parents are ready to be scheduled if Direction=TopDown.
  Nodes without children are ready to be scheduled if Direction=BottomUp.
******************************************************************************/
struct SchedDagNode {
  SchedDagNode(Operation *op) : op(op) {
    static int32_t serialId = 0;
    id = serialId++;
    opStr = opStringShort(op);
  }

  // Copy constructor.
  SchedDagNode(const SchedDagNode& node) : op(node.op), id(node.id), opStr(node.opStr), children(node.children), parents(node.parents) {}

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
  template <SchedDirection Direction>
  bool isReady() {
    if constexpr (Direction == SchedDirection::TopDown) {
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

  // returns true of this is ancestor (by recursively checking children) of other.
  bool isAncestor(SchedDagNode *other) const {
    for (SchedDagNode *child : children) {
      if (child == other) {
        // this is ancestor of other if other is my child
        return true;
      } else if (child->isAncestor(other)) {
        // this is ancestor of other if one of my children are ancestor of other.
        return true;
      }
    }
    return false;
  }

  // returns true of this is posterity (by recursively checking parents) of other.
  bool isPosterity(SchedDagNode *other) const {
    for (SchedDagNode *parent : parents) {
      if (parent == other) {
        // this is posterity of other if other is my parent
        return true;
      } else if (parent->isPosterity(other)) {
        // this is posterity of other if one of its parents are posterity of other.
        return true;
      }
    }
    return false;
  }


  Operation *op;
  int32_t id; // for debug printing
  std::string opStr;

  // Children depend on this node; this node must be scheduled before children.
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

llvm::raw_ostream &operator<<(llvm::raw_ostream &out, SchedDagNodeList &nodes) {
  for (auto node : nodes) {
    out << *node << "\n";
  }
  return out;
}

llvm::raw_ostream &dumpDagDotFormat(llvm::raw_ostream &out, SchedDagNodeList &nodes) {
  out << "digraph \"dep-dag\" {\n";
  out << "rankdir=\"BT\"\n";
  for (auto node : nodes) {
    std::string name = std::to_string(reinterpret_cast<intptr_t>(node));
    out << name << "\t[label=\"[@" << node->id << " " << node->opStr << "]\"]\n";
  }
  for (auto node : nodes) {
    std::string name = std::to_string(reinterpret_cast<intptr_t>(node));
    for (auto child : node->getChildren()) {
      std::string childName = std::to_string(reinterpret_cast<intptr_t>(child));
      out << "\t" << childName << " -> " << name << ";\n";
    }
  }
  out << "}\n";
  return out;
}

llvm::raw_ostream &dumpDagDotFormat(llvm::raw_ostream &out, llvm::SmallVector<std::shared_ptr<SchedDagNode>> &nodes) {
  out << "digraph \"dep-dag\" {\n";
  out << "rankdir=\"BT\"\n";
  for (auto node : nodes) {
    std::string name = std::to_string(reinterpret_cast<intptr_t>(node.get()));
    out << name << "\t[label=\"[@" << node.get()->id << " " << node.get()->opStr << "]\"]\n";
  }
  for (auto node : nodes) {
    std::string name = std::to_string(reinterpret_cast<intptr_t>(node.get()));
    for (auto child : node->getChildren()) {
      std::string childName = std::to_string(reinterpret_cast<intptr_t>(child));
      out << "\t" << childName << " -> " << name << ";\n";
    }
  }
  out << "}\n";
  return out;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &out, llvm::SmallVector<std::shared_ptr<SchedDagNode>> &nodes) {
  for (auto node : nodes) {
    out << *(node.get()) << "\n";
  }
  return out;
}

bool opCategoryLoad(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<
      triton::LoadOp,
      triton::gpu::LocalLoadOp,
      triton::amdgpu::BufferLoadOp
    >(op);
}
bool opCategoryStore(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<
  triton::StoreOp,
  triton::gpu::LocalStoreOp
    >(op);
}
bool opCategoryMem(SchedDagNode *node) {
  return opCategoryLoad(node) || opCategoryStore(node);
}

bool opCategoryGlobalLoad(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<
      triton::LoadOp,
      triton::amdgpu::BufferLoadOp
    >(op);
}
bool opCategoryGlobalStore(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<
  triton::StoreOp
    >(op);
}

bool opCategoryNop(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<
      triton::gpu::MemDescSubviewOp,
      triton::gpu::MemDescTransOp,
      tt::amdgpu::ExtractSliceOp,
      tt::amdgpu::ConcatOp
    >(op);
}

bool opCategoryBarrier(SchedDagNode *node) {
  Operation *op = node->getOp();
  return llvm::isa<
  mlir::gpu::BarrierOp
  // sched.barrier
  // setprio
    >(op);
}


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

/******************************************************************************
 Dependency is from src/child to dst/parent.
 dst must preceed src during scheduling.
 dst / parent
  ^
  |
 src / child
******************************************************************************/
struct SchedDep {
  SchedDagNode *parent; // dst
  SchedDagNode *child; // src
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &out, SchedDep &dep) {
  return  out << *(dep.child) << " -> " << *(dep.parent);
}

typedef SmallVector<SchedDep> DepList;
typedef DenseMap<StringRef, DepList> DepMap;

/******************************************************************************
  Abstract base class.
  calcDeps() gets to see the nodeList with all currently known deps applied to nodes.
******************************************************************************/
struct DependencyCalculator {
  DependencyCalculator(StringRef name) : depName(name) {}

  virtual void calcDeps() = 0;

  DepList calcDepsForNodes(SchedDagNodeList& currentNodeList, BasicBlockNodeMap currentNodeMap) {
    LDBG("DependencyCalculator::calcDepsForNodes() - " << depName);
    depList.clear();
    nodeList = &currentNodeList;
    nodeMap = &currentNodeMap;
    calcDeps();
    return depList;
  }
  virtual ~DependencyCalculator() = default;

  StringRef depName;
  SchedDagNodeList *nodeList;
  DepList depList;
  BasicBlockNodeMap *nodeMap;
};


/******************************************************************************
  Create data dependencies based on def-use chains.
  Also creates dependencies based on various barriers.
  This class represents the minimum set of dependencies needed for correctness.
******************************************************************************/
struct DataDependencyCalculator : DependencyCalculator {
  DataDependencyCalculator() :
      DependencyCalculator("Data") {}

  // There is an implied data dependency between LDS ops and GPUBarrier.
  template <SchedDirection Direction> void calcDepsLdsGpuBar() {

    auto fwIt = nodeList->begin();
    auto bkIt = nodeList->rbegin();
    auto next = [&]() -> SchedDagNode * {
      if constexpr (Direction == SchedDirection::TopDown) {
        if (fwIt == nodeList->end())
          return nullptr;
        return *(fwIt++);
      }
      if constexpr (Direction == SchedDirection::BottomUp) {
        if (bkIt == nodeList->rend())
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
          if constexpr (Direction == SchedDirection::TopDown) {
            SchedDep dep;
            dep.parent = ldsOpNode;
            dep.child = barrierNode;
            depList.push_back(dep);
          }
          if constexpr (Direction == SchedDirection::BottomUp) {
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
    for (auto it = std::next(nodeList->begin()); it != nodeList->end(); ++it) {
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
    for (auto it = nodeList->begin(); it != nodeList->end(); ++it) {
      SchedDagNode *node = (*it);
      for (auto operandValue : node->getOp()->getOperands()) {
        auto operandDefOp = operandValue.getDefiningOp();
        SchedDagNode *parentNode = (*nodeMap)[operandDefOp];
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
    SchedDagNode *lastNode = (*(nodeList->rbegin())); // cf.br
    for (auto it = std::next(nodeList->rbegin()); it != nodeList->rend(); ++it) {
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
    calcDepsOperands();
    calcDepsLdsGpuBar<SchedDirection::BottomUp>();
    calcDepsLdsGpuBar<SchedDirection::TopDown>();
    calcDepsGpuBarGpuBar();
    calcDepsCfBr();
    calcDepsSetPrio();
  }

};

/******************************************************************************
  Create order dependencies between ops which were refined from same original op.
******************************************************************************/
struct RefinedOpDependencyCalculator : DependencyCalculator {
  RefinedOpDependencyCalculator() :
      DependencyCalculator("RefinedOrder") {}

  /*
    Add dependencies between all dots; it is assumed that they were originally
    created in the ideal order during refinement.
    If it changes that they aren't created in ideal order, then we would need to enhance this
    to place dependencies according to dot tiling.
  */
  void calcDepsDot() {
    SchedDagNode *prevDot = nullptr;
    int32_t prevTileSerial = -1;
    int32_t prevElemSerial = -1;
    for (auto it = nodeList->begin(); it != nodeList->end(); ++it) {
      SchedDagNode *node = *it;
      if (DotOp op = dyn_cast<DotOp>(node->getOp())) {
        if (op->hasAttr(triton::amdgpu::DotTileAttr::getMnemonic())) {
          auto attr = op->getAttrOfType<triton::amdgpu::DotTileAttr>(triton::amdgpu::DotTileAttr::getMnemonic());
          //int32_t tileSerial = attr.getTileSerial();
          //int32_t elemSerial = attr.getElementSerial();
          //LDBG("Found DotOp w/ DotTileAttr " << tileSerial << ", " << elemSerial);
          // When both tile and element serial ids decrease, it means we've started a new dot.
          if (prevDot /*&& (tileSerial >= prevTileSerial || elemSerial >= prevElemSerial)*/) {
            SchedDep dep;
            dep.parent = prevDot;
            dep.child = node;
            depList.push_back(dep);
          }
          prevDot = node;
          //prevTileSerial = tileSerial;
          //prevElemSerial = elemSerial;
        } else {
          LDBG("WARNING DotOp has no DotTileAttr:" << *node);
        }
      }
    }
  }

  /*
    Add dependencies between local loads which
    (1) Belong to the same dot.
    (2) Belong to the same operand of the dot.
    It is assumed that the above two criterial mean the local loads would have been refined
    from the same unrefined local load.
    It is also assumed that the ideal relative order of refined local loads
    is monotonically increasing addresses.
  */
  void calcDepsLocalLoad() {
    SchedDagNode *prevNode = nullptr;
    int32_t prevTileSerial = -1;
    int32_t prevElemSerial = -1;
    int32_t prevOpdIdx = -1;
    for (auto it = nodeList->begin(); it != nodeList->end(); ++it) {
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
    Add dependencies between ops which were refined from the same op.
    It is assumed that the ideal relative order of refines ops is the order created during refinement.
  */
  void calcDepsRefinedOp() {
    SchedDagNode *prevNode = nullptr;
    int32_t prevId = -1;
    for (auto it = nodeList->begin(); it != nodeList->end(); ++it) {
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

};

/*
  Simple DependencyCalculators based on op types (or category) and serializes all ops of that type.
*/
template<typename OpType>
void calcDepsOpType(SchedDagNodeList *nodeList, DepList& depList) {
  SchedDagNode *prevNode = nullptr;
  int32_t prevId = -1;
  for (auto it = nodeList->begin(); it != nodeList->end(); ++it) {
    SchedDagNode *node = *it;
    if (llvm::isa<OpType>(node->getOp())) {
      if (prevNode) {
        SchedDep dep;
        dep.parent = prevNode;
        dep.child = node;
        depList.push_back(dep);
      }
      prevNode = node;
    }
  }
}

void calcDepsOpCategory(SchedDagNodeList *nodeList, DepList& depList, bool (*category)(SchedDagNode *)) {
  SchedDagNode *prevNode = nullptr;
  int32_t prevId = -1;
  for (auto it = nodeList->begin(); it != nodeList->end(); ++it) {
    SchedDagNode *node = *it;
    if (category(node)) {
      if (prevNode) {
        SchedDep dep;
        dep.parent = prevNode;
        dep.child = node;
        depList.push_back(dep);
      }
      prevNode = node;
    }
  }
}

struct LocalLoadOrderDependencyCalculator : DependencyCalculator {
  LocalLoadOrderDependencyCalculator() :
      DependencyCalculator("LocalLoadTypeOrder") {}
  void calcDeps() {
    calcDepsOpType<triton::gpu::LocalLoadOp>(nodeList, depList);
  }
};

struct LocalStoreOrderDependencyCalculator : DependencyCalculator {
  LocalStoreOrderDependencyCalculator() :
      DependencyCalculator("LocalStoreTypeOrder") {}
  void calcDeps() {
    calcDepsOpType<triton::gpu::LocalStoreOp>(nodeList, depList);
  }
};

struct GlobalLoadOrderDependencyCalculator : DependencyCalculator {
  GlobalLoadOrderDependencyCalculator() :
      DependencyCalculator("GlobalLoadCategoryOrder") {}
  void calcDeps() {
    calcDepsOpCategory(nodeList, depList, opCategoryGlobalLoad);
  }
};


/******************************************************************************
  SchedDag consists of nodes containing edges to other nodes.
  DepMap stored outside of Dag.
******************************************************************************/
struct SchedDag {
public:
  SchedDag(SchedDagNodeList nodes) : nodes(nodes) {}

  SmallVector<SchedDagNode *> getNodes() {
    SmallVector<SchedDagNode *> copy(nodes.size(), nullptr);
    for (auto [idx, node] : llvm::enumerate(nodes)) {
      copy[idx] = node;
    }
    return copy;
  }

  void addDeps(const DepMap & deps) {
    for (auto& depType : deps) {
      StringRef depTypeName = depType.getFirst();
      DepList depList = depType.getSecond();
      for (auto dep : depList) {
        dep.child->addParent(dep.parent);
        dep.parent->addChild(dep.child);
      }
    }
  }

  void addDeps(const DepList & depList) {
    for (auto dep : depList) {
      dep.child->addParent(dep.parent);
      dep.parent->addChild(dep.child);
    }
  }

  void resetDeps() {
    for (auto [idx, node] : llvm::enumerate(nodes)) {
      node->clearDeps();
    }
  }

  template <SchedDirection Direction>
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
  template <SchedDirection Direction>
  void removeScheduledNode(SchedDagNode *node) {
    assert(node->isReady<Direction>());
    readyNodes.remove(node);

    if constexpr (Direction == SchedDirection::TopDown) {
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

/******************************************************************************
Create high-level dependencies between the different memory ops where there aren't data deps.
A basic block has a collection of refined memory ops, GL, LS, LL.

for num_stages=2 without local_prefetch
the GL and LS have data dependencies, and both can be co-scheduled with the local_loads.
So we need to specify that the GL should overlap the LL. The LS can't overlap LL because of anti-dep enforced by gpu.bar.

Example:
GLA01
GLB01
LLA0123
LLB0123
LSA01
LSB01

Could be serialized like

GLA01 GLB01 LLA0123 LLB0123
LSA01 LSB01

or

GLA01 GLB01 LLA0123 LLB0123
LSA01 LSB01

https://github.com/ROCm/triton-internal/issues/736
Determine Memory Op Order & Co-Scheduling
We need to determine the relative ordering between or overlap of different memory op types; e.g. if global_loads and local_loads can both be scheduled, which should have higher priority. The ordering of memory ops have implications for performance and register use. This applies to loops where data can be prefetched, and also to regions with multiple dots.
First, data dependencies determine order.
Second, critical-path analysis determines order, meaning the memory ops which need to be scheduled to get to the mfmas sooner have highest priority.
Note that for local_loads, some may be on the critical path to get us to the first 1-2 dot-tiles, but after other dot-tiles' local_loads can be delayed
Third, based on scheduling mode:
Min-Vgpr mode: preferred order is local_store, global_load, local_load.
Max-Latency mode: preferred order is global_load, local_load, local_store.
========================================================================
This applies to loops where data can be prefetched.
This should also apply to regions with multiple dots.
First, def-use dependencies kick in to determine order
If LS overlaps LL (both prefetched, 011, 111)
  min-vgpr: LS, LL to save vgprs
  max-latency: LL, LS to maximize GL latency hiding
  ever want to co-schedule? - no
If GL overlaps LL (GL and (LS or LL) prefetched; 101, 110, 111)
  min-vgpr: GL, LL
  max-latency: GL, LL (same)
If GL overlaps LS
  min-vgpr: LS before GL to save vgprs
  max-latency: GL before LS to maximize latency hiding
If GL overlaps LS overlaps LL
  min-vgpr: LS, GL, LL to save vgprs
  max-latency: GL, LL, LS
Need to verify the above doesn’t create any cyclic dependencies?
Need to clarify that some of these “after” are actually “interleave”.
Need to distinguish between vgpr vs latency for global separate from LDS ops?

The above logic is hyper-focused on gemm, and we need to abstract much more.

==========================================================================
This is actually mostly determined during StreamPipeliner via clustering and num_stages. We need to decide the optimal memoy op ordering (via cluster) during StreamPipeliner and covey it forward via the IR.
Additionally, we also need StreamPipeliner to specify whether op A should fully preceed op B, or whether op A and op B can overlap and therefore be interleaved after refining.
==========================================================================



This first issue has mostly to do with when everything is prefetched, what order should they be in.



AddDeps: Order deps between different memory types #738
https://github.com/ROCm/triton-internal/issues/738
Memory Opr Order Deps

With this ordering of ops, we can create order dependencies between op types so we only schedule the ones at a time that we want. E.g. for local prefetch, scheduler shouldn’t even see local_loads until very late in the loop.

Insert order dependencies to serialize certain memory ops.

This may need information to be communicated from the stream-pipeliner (or other places) to the scheduler. Some examples of what this task means:
(1) A gemm with num_stages=2 wants local_loads close to the top of the kernel and global_loads to come after the first few local_loads but interleaved among later local_loads. Whereas enabling local_prefetch puts all the local_loads at the bottom of the loop after the global_loads and local_stores. The scheduler needs to know "delay the local_loads as much as possible" after most mfmas which have freed registers which the local_local loads can then use. The backend gets this wrong because it schedules the local_loads early and uses too many vgprs.

(2) For FA num_stages=2 maxDepth=1, the memory structure looks like
global_load
local_store <-- these must come before
global_load <-- these to save vgprs
local_store

This second issue takes

GLA01 GLB01 LLA0123 LLB0123
LSA01 LSB01

and further narrows it to 

LLA0
LLB0
LLB1
LLA1
GLA0
LLA2
GLA1
LLB2
GLB0
LLB3
GLB1
LLA3
LSA0
LSA1
LSB0
LSB1

Later will AddDeps between the mem ops and 


global_load before/after local_loads
global_loads before/after local_stores
local_load[A] before/after local_load[B]
global_load[K] before/after global_load[K]


Determines which memory ops should overlap other memory ops (vs being co-scheduled)
when data dependencies allow them to be.

if we have all dots in order, and we do a scheduling with delaying all local loads
then we can just grab the order of local_loads from that.
Can we just keep the relative order of local stores as being final?
Then have the relative order of global loads to match (with loop wrap around).
******************************************************************************/
struct MemOrderDependencyCalculator : DependencyCalculator {
  MemOrderDependencyCalculator() :
      DependencyCalculator("MemOrder") {}

  // get memory ops only from the graph; keep them in order.
  llvm::SmallVector<std::shared_ptr<SchedDagNode>> getMemNodes() const {
    llvm::SmallVector<std::shared_ptr<SchedDagNode>> memNodes;
    for (SchedDagNode *node : *nodeList) {
      if (opCategoryMem(node)) {
        // Verify unrefined op not already in list.
        Operation *op = node->getOp();
        if (op->hasAttr(triton::amdgpu::RefinedOpOrderAttr::getMnemonic())) {
          auto attr = op->getAttrOfType<triton::amdgpu::RefinedOpOrderAttr>(triton::amdgpu::RefinedOpOrderAttr::getMnemonic());
          int32_t idOriginalOp = attr.getIdOriginalOp();
          int32_t idRefinedOp = attr.getIdRefinedOp();
          if (idRefinedOp == 0) {
            LDBG("Found MemNode: " << *node);
            // create a copy of the new node, since we don't want to disturb the old graph's deps.
            std::shared_ptr<SchedDagNode> nodeClone = std::make_shared<SchedDagNode>(*node);
            nodeClone.get()->clearDeps();
            memNodes.push_back(nodeClone);
          }
        } else {
          llvm::dbgs() << "Found memory op without RefinedOpOrderAttr: " << op << "\n";
        }
      }
    }

    // Add edges directly into simplified memory graph.
    for (auto it0 = memNodes.begin(); it0 != memNodes.end(); ++it0) {
      SchedDagNode *memNode0 = it0->get();
      Operation *memOp0 = memNode0->getOp();
      SchedDagNode *origNode0 = (*nodeMap)[memOp0];
      for (auto it1 = memNodes.begin(); it1 != memNodes.end(); ++it1) {
        SchedDagNode *memNode1 = it1->get();
        Operation *memOp1 = memNode1->getOp();
        SchedDagNode *origNode1 = (*nodeMap)[memOp1];
        if (origNode0->isAncestor(origNode1)) {
          // node0 is ancestor of node1 therefore add dependencies.
          memNode0->addChild(memNode1);
          memNode1->addParent(memNode0);
        }
      }
    }
    return memNodes;
  }
    
  /*
  Need to traverse graph to determine which memory ops can be co-scheduled with which other memory ops.

  aiter gemm has 2 decisions to be made
  Should buffer load come before/after local_loads.

  */
  void calcDeps() {
    llvm::SmallVector<std::shared_ptr<SchedDagNode>> memNodes = getMemNodes();
    LDBG("Simplified Graph of MemNodes");
    LDBG(memNodes);
    dumpDagDotFormat(llvm::dbgs(), memNodes);

    // If memory ops' data are used in strict order, then 

  }

};

/******************************************************************************
  Library of comparison functions for scheduling heuristics.
******************************************************************************/

// Prefer op based on type.
template <typename OpType>
bool preferOpType(SchedDagNode *a, SchedDagNode *b, bool prefer = true) {
  return (llvm::isa<OpType>(a->getOp()) and !llvm::isa<OpType>(b->getOp())) == prefer;
}

template <typename OpType>
SchedDagNode *findPreferredOpType(SchedDagNode *a, SchedDagNode *b, bool prefer = true) {
  if (preferOpType<OpType>(a, b, prefer)) {
    return a;
  } else if (preferOpType<OpType>(b, a, prefer)) {
    return b;
  }
  return nullptr;
}

// Prefer op based on category (e.g. opCategoryLoad()).
bool preferOpCategory(SchedDagNode *a, SchedDagNode *b, bool (*category)(SchedDagNode *), bool prefer = true) {
  return (category(a) and !category(b)) == prefer;
}

SchedDagNode *findPreferredOpCategory(SchedDagNode *a, SchedDagNode *b, bool (*category)(SchedDagNode *), bool prefer = true) {
  if (preferOpCategory(a, b, category, prefer)) {
    return a;
  } else if (preferOpCategory(b, a, category, prefer)) {
    return b;
  }
  return nullptr;
}

bool preferLocalLoad(SchedDagNode *a, SchedDagNode *b, bool prefer = true) {
  return (llvm::isa<triton::gpu::LocalLoadOp>(a->getOp()) and !llvm::isa<triton::gpu::LocalLoadOp>(b->getOp())) == prefer;
}

// Returns ops in original mlir block order.
template<SchedDirection Direction>
SchedDagNode *getOriginalOrder(SchedDagNode *a, SchedDagNode *b) {
  return (a->id < b->id) == (Direction == SchedDirection::TopDown) ? a : b;
}

/*
  Abstract Base class for a scheduling heuristic recipe.
  E.g. TopDown, schedule global_loads early and local_stores late.
*/
template<SchedDirection Direction>
struct SchedHeuristic {
  virtual SchedDagNode *operator()(SchedDagNode *a, SchedDagNode *b) = 0;
  virtual StringRef name() = 0;
};

/******************************************************************************
  Delay LocalLoads as much as possible so they're adjacent to the dot which needs them.
  This will then allow for placing deps between local loads.
  Also delay other memory ops so that order deps can be placed between them too.
  SchedDirection = BottomUp
******************************************************************************/
struct SchedHeuristicLocalLoadAdjacentDotOps : public SchedHeuristic<SchedDirection::BottomUp> {

  SchedDagNode *operator()(SchedDagNode *a, SchedDagNode *b) {
    // Schedule local loads asap.
    if (auto selected = findPreferredOpType<triton::gpu::LocalLoadOp>(a, b)) {
      return selected;
    }
    // Schedule others that could be in the way of local loads.
    if (auto selected = findPreferredOpCategory(a, b, opCategoryNop)) {
      return selected;
    }
    if (auto selected = findPreferredOpCategory(a, b, opCategoryBarrier)) {
      return selected;
    }
    if (auto selected = findPreferredOpType<triton::gpu::LocalStoreOp>(a, b)) {
      return selected;
    }

    // Also delay other memory ops.
    if (auto selected = findPreferredOpCategory(a, b, opCategoryGlobalStore)) {
      return selected;
    }
    if (auto selected = findPreferredOpCategory(a, b, opCategoryGlobalLoad)) {
      return selected;
    }

    // Schedule non-dots, so that all non-dot dependencies are fulfilled
    // to better guarantee that local loads will be adjacent to dots.
    if (auto selected = findPreferredOpType<triton::DotOp>(a, b, false)) {
      return selected;
    }

    // Final comparison based on orig order.
    return getOriginalOrder<SchedDirection::BottomUp>(a, b);
  }

  StringRef name() { return "LocalLoadAdjacentDot"; }
};

/******************************************************************************
  Schedule original order (insomuch as dependencies allowed).
  Also hoist nops (which groups together) for ease of reading.
******************************************************************************/
struct SchedHeuristicOriginalOrder : public SchedHeuristic<SchedDirection::TopDown> {

  SchedDagNode *operator()(SchedDagNode *a, SchedDagNode *b) {
    if (auto selected = findPreferredOpCategory(a, b, opCategoryNop)) {
      return selected;
    }
    return getOriginalOrder<SchedDirection::TopDown>(a, b);
  }

  StringRef name() { return "OriginalOrder"; }
};

/******************************************************************************
  SchedManager
  - Tracks dependencies.
  - Prepared dag for scheduling.
  - Runs scheduler.
******************************************************************************/
struct SchedManager {
  SchedManager(Block *block) :
      nodeMap(block),
      nodeList(nodeMap.getNodeList()),
      dag(nodeList),
      rescheduleId(0) {}

  // Calculate new deps based on op order and previously determined deps.
  // Insert new deps into dep map and apply them to dat.
  void addDeps(std::unique_ptr<DependencyCalculator> depCalc) {
    deps[depCalc->depName] = depCalc->calcDepsForNodes(nodeList, nodeMap);
    dag.addDeps(deps[depCalc->depName]);
    dumpDagDotFormat(llvm::dbgs(), nodeList);
  }

  void printNodes() {
    LDBG("nodeList:");
    for (auto it = nodeList.begin(); it != nodeList.end(); ++it) {
      auto node = *it;
      LDBG(*node);
    }
  }

  template<SchedDirection Direction>
  void reschedule(SchedHeuristic<Direction> *heuristic) {
    LDBG("");
    LDBG(hr);
    LDBG("SchedManager::reschedule(" << rescheduleId << "), Direction="
        << ((Direction == SchedDirection::TopDown) ? "TopDown" : "BottomUp")
        << ", Heuristic=" << heuristic->name()
    );
    LDBG(hr);

    LDBG("NodeList before reschedule(" << rescheduleId << ")");
    LLVM_DEBUG(printNodes());

    // Node readiness is based on direction.
    dag.initReadyNodes<Direction>();

    // Schedule the dag; this process removes deps from nodes.
    LDBG("");
    LDBG(hr);
    // Store nodes in newly scheduled order.
    SchedDagNodeList rescheduledNodes;
    const bool printDetails = false;
    for (int iter = 0; !dag.finished(); ++iter) {
      const auto &readyNodes = dag.getReadyNodes();
      SchedDagNode *selectedNode = selectFromReadyNodes(readyNodes, heuristic);
      if (printDetails) {
        std::string dbgStr;
        llvm::raw_string_ostream dbgStream(dbgStr);
        dbgStream << "Iter: " << iter << "\n";
        dbgStream << "Ready List:\n";
        for (auto node : readyNodes) {
          dbgStream << "    " << *node << "\n";
        }
        dbgStream << "reschedule(" << rescheduleId << ") Selected: " << *selectedNode << "\n";
        LDBG(dbgStream.str());
      }
      // Place selected node in list, remove it from dag which updates readyList.
      rescheduledNodes.push_back(selectedNode);
      dag.removeScheduledNode<Direction>(selectedNode);
    }

    // After scheduling, re-apply deps to prepare for adding additional deps.
    dag.addDeps(deps);

    // Update nodeList after rescheduling.
    if constexpr (Direction == SchedDirection::TopDown) {
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

  template<SchedDirection Direction>
  SchedDagNode *selectFromReadyNodes(SetVector<SchedDagNode *> readyNodes, SchedHeuristic<Direction> *heuristic) {
    SchedDagNode *selected = readyNodes.front();
    for (auto it = std::next(readyNodes.begin()); it != readyNodes.end(); ++it) {
      SchedDagNode *node = *it;
      selected = (*heuristic)(selected, node);
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
  bool readyToAddDeps;
  DepMap deps;
  DenseMap<SchedDagNode *, size_t> nodeIndices;
  int32_t rescheduleId;
}; // SchedManager

/******************************************************************************
  TritonAMDGPURescheduleOps::applyReschedulingPasses()
  is the top-level scheduling pass for a single block,
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
******************************************************************************/
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

  void applyReschedulingPasses(Block *mlirBlock) {
    LDBG("");
    LDBG("");
    LDBG("TritonAMDGPURescheduleOps::applyReschedulingPasses()");

    SchedManager schedManager(mlirBlock);

    // (0) Add basic deps.
    // DataDependencyCalculator dataDepCalc;
    schedManager.addDeps(std::make_unique<DataDependencyCalculator>());
    schedManager.addDeps(std::make_unique<RefinedOpDependencyCalculator>());

    // (1) Considering mfmas are in optimal order, find optimal local_load order.
    SchedHeuristicLocalLoadAdjacentDotOps sh0;
    schedManager.reschedule<SchedDirection::BottomUp>(&sh0);
    schedManager.addDeps(std::make_unique<LocalLoadOrderDependencyCalculator>());
    schedManager.addDeps(std::make_unique<LocalStoreOrderDependencyCalculator>());
    schedManager.addDeps(std::make_unique<GlobalLoadOrderDependencyCalculator>());
    schedManager.addDeps(std::make_unique<MemOrderDependencyCalculator>());

    // (F) Final rescheduling restores original order except for dependencies.
    SchedHeuristicOriginalOrder shF;
    schedManager.reschedule<SchedDirection::TopDown>(&shF);

    SmallVector<Operation *> rescheduledOps = schedManager.getOpList();

    // Print op (and not node) list.
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

    // Apply schedule to basic block.
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
