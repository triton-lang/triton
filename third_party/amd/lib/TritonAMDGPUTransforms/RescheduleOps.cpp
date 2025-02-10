#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
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

namespace {
struct Node {
  Node(Operation *op) : op(op) {}
  enum class ChildType { Real, Artificials };

  Operation *getOp() { return op; }

  template <ChildType Type> void add(Node *node) {
    llvm::SetVector<Node *> *children =
        Type == ChildType::Real ? &realChildren : &artificialChildren;
    children->insert(node);
  }
  void addParent(Node *node) { parents.insert(node); }

  size_t getNumParents() { return parents.size(); }
  bool hasChildren() {
    return !(realChildren.empty() && artificialChildren.empty());
  }
  bool hasNoChildren() { return !hasChildren(); }

  const llvm::SetVector<Node *> &getRealChildren() { return realChildren; }
  const llvm::SetVector<Node *> &getArtificialChildren() {
    return artificialChildren;
  }
  const llvm::SetVector<Node *> &getParents() { return parents; }

  void removeChild(Node *node) {
    if (realChildren.contains(node)) {
      realChildren.remove(node);
    }
    if (artificialChildren.contains(node)) {
      artificialChildren.remove(node);
    }
  }

  void removeParent(Node *node) {
    if (parents.contains(node)) {
      parents.remove(node);
    }
  }

  void drainChildren() {
    realChildren.clear();
    artificialChildren.clear();
  }

private:
  Operation *op;
  llvm::SetVector<Node *> realChildren;
  llvm::SetVector<Node *> artificialChildren;
  llvm::SetVector<Node *> parents;
};

struct Graph {
public:
  Graph(Block *mlirBlock) {
    createNodes(mlirBlock);
    createEdges();
  }

  Graph(const Graph &other) {
    using iteratorType = decltype(other.nodes.begin());
    DenseMap<Node *, iteratorType> map;
    for (auto it = other.nodes.begin(); it != other.nodes.end(); ++it) {
      auto newNode = std::make_unique<Node>(it->get()->getOp());
      map.insert({it->get(), it});
      lookup.insert({newNode->getOp(), newNode.get()});
      nodes.push_back(std::move(newNode));
    }

    for (auto [idx, otherNode] : llvm::enumerate(other.nodes)) {
      auto &currNode = nodes[idx];
      for (auto otherChild : otherNode->getRealChildren()) {
        auto otherChildIt = map.find(otherChild)->second;
        auto childIdx = std::distance(other.nodes.begin(), otherChildIt);
        currNode->add<Node::ChildType::Real>(nodes[childIdx].get());
      }

      for (auto otherChild : otherNode->getArtificialChildren()) {
        auto otherChildIt = map.find(otherChild)->second;
        auto childIdx = std::distance(other.nodes.begin(), otherChildIt);
        currNode->add<Node::ChildType::Artificials>(nodes[childIdx].get());
      }

      for (auto otherParent : otherNode->getParents()) {
        auto otherParentIt = map.find(otherParent)->second;
        auto parentIdx = std::distance(other.nodes.begin(), otherParentIt);
        currNode->addParent(nodes[parentIdx].get());
      }
      nodes.push_back(std::make_unique<Node>(otherNode->getOp()));
    }
  }

  SmallVector<Node *> getNodes() {
    SmallVector<Node *> copy(nodes.size(), nullptr);
    for (auto [idx, node] : llvm::enumerate(nodes)) {
      copy[idx] = node.get();
    }
    return copy;
  }

private:
  void createNodes(Block *mlirBlock) {
    for (auto it = mlirBlock->begin(); it != mlirBlock->end(); ++it) {
      Operation *op = &(*it);
      std::unique_ptr<Node> node = std::make_unique<Node>(op);
      lookup.insert({op, node.get()});
      nodes.push_back(std::move(node));
    }
  }

  enum class Traversal { Topdown, Bottomup };
  template <Traversal Direction> void insertGPUBarrierEdges() {

    auto fwIt = nodes.begin();
    auto bkIt = nodes.rbegin();
    auto next = [&]() -> Node * {
      if constexpr (Direction == Traversal::Topdown) {
        if (fwIt == nodes.end())
          return nullptr;
        return (fwIt++)->get();
      }
      if constexpr (Direction == Traversal::Bottomup) {
        if (bkIt == nodes.rend())
          return nullptr;
        return (bkIt++)->get();
      }
      return nullptr;
    };

    llvm::SmallVector<Node *> ldsOpsNodes;
    while (Node *node = next()) {
      auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(node->getOp());
      auto localStore =
          llvm::dyn_cast<triton::gpu::LocalStoreOp>(node->getOp());
      if (localLoad || localStore) {
        ldsOpsNodes.push_back(node);
      }
      auto gpuBarrier = llvm::dyn_cast<mlir::gpu::BarrierOp>(node->getOp());
      if (gpuBarrier) {
        Node *barrierNode = node;
        for (auto ldsOpNode : ldsOpsNodes) {
          if constexpr (Direction == Traversal::Topdown) {
            barrierNode->add<Node::ChildType::Artificials>(ldsOpNode);
            ldsOpNode->addParent(barrierNode);
          }
          if constexpr (Direction == Traversal::Bottomup) {
            barrierNode->addParent(ldsOpNode);
            ldsOpNode->add<Node::ChildType::Artificials>(barrierNode);
          }
        }
        ldsOpsNodes.clear();
      }
    }
  }

  void createEdges() {
    // insert edges imposed by def-use chains
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
      auto &node = *it;
      for (auto operandValue : node->getOp()->getOperands()) {
        auto operandDefOp = operandValue.getDefiningOp();
        if (!lookup.contains(operandDefOp))
          continue;
        Node *childNode = lookup.find(operandDefOp)->second;
        node->add<Node::ChildType::Real>(childNode);
        childNode->addParent(node.get());
      }
    }

    // gpu.Barrier ops are orphans. Add edges to
    // respect data dependencies in the block
    insertGPUBarrierEdges<Traversal::Bottomup>();
    insertGPUBarrierEdges<Traversal::Topdown>();

    // connect orphans with the last op in the block
    auto &lastNode = *(nodes.rbegin());
    for (auto it = std::next(nodes.rbegin()); it != nodes.rend(); ++it) {
      auto &node = *it;
      if (node->getNumParents() == 0) {
        node->addParent(lastNode.get());
        lastNode->add<Node::ChildType::Artificials>(node.get());
      }
    }
  }

  llvm::SmallVector<std::unique_ptr<Node>> nodes;
  llvm::MapVector<Operation *, Node *> lookup;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &out, Graph &graph) {
  out << "digraph \"dep-graph\" {\n";
  out << "rankdir=\"LR\"\n";
  for (auto [idx, node] : llvm::enumerate(graph.getNodes())) {
    std::string name = std::to_string(reinterpret_cast<intptr_t>(node));
    out << name << "\t[label=\"" << node->getOp()->getName() << "\"]\n";
  }
  for (auto [idx, node] : llvm::enumerate(graph.getNodes())) {
    std::string name = std::to_string(reinterpret_cast<intptr_t>(node));
    for (auto child : node->getRealChildren()) {
      std::string childName = std::to_string(reinterpret_cast<intptr_t>(child));
      out << "\t" << childName << " -> " << name << ";\n";
    }
    for (auto child : node->getArtificialChildren()) {
      std::string childName = std::to_string(reinterpret_cast<intptr_t>(child));
      out << "\t" << childName << " -> " << name
          << " [style=\"dashed\", color=\"blue\"];\n";
    }
  }
  out << "}";
  return out;
}

struct GraphManager {
  GraphManager(Graph &graph) : graph(graph) {
    for (auto [idx, node] : llvm::enumerate(graph.getNodes())) {
      nodesIndices.insert({node, idx});
      if (node->hasNoChildren())
        leafs.insert(node);
    }
  }

  bool finished() { return leafs.empty(); }

  void removeLeaf(Node *node) {
    assert(node->hasNoChildren());
    leafs.remove(node);
    for (auto parent : node->getParents()) {
      parent->removeChild(node);
      if (parent->hasNoChildren()) {
        leafs.insert(parent);
      }
    }
  }

  size_t getNodeSourceCodeIndex(Node *node) {
    assert(nodesIndices.contains(node));
    return nodesIndices[node];
  }

  const SetVector<Node *> &getCurrentLeafs() { return leafs; }

private:
  Graph graph;
  SetVector<Node *> leafs;
  DenseMap<Node *, size_t> nodesIndices;
};

struct MachineModel {
  struct Result {
    Node *selectedNode{nullptr};
    SmallVector<Node *> normPriorityNodes{};
    SmallVector<Node *> lowPriorityNodes{};
    void set(Node *node) {
      if (!selectedNode)
        selectedNode = node;
    }
  };

  Result select(const SetVector<Node *> &readyNodes) {
    Result result;
    for (auto *node : readyNodes) {
      Operation *op = node->getOp();
      if (dyn_cast<triton::gpu::LocalLoadOp>(op) ||
          dyn_cast<triton::gpu::LocalStoreOp>(op)) {
        if (MachineModel::maxLocalLoadStoreIssues >
            issuedLocalStoreLoadCounter) {
          ++issuedLocalStoreLoadCounter;
          result.set(node);
        } else {
          result.lowPriorityNodes.push_back(node);
        }
        continue;
      }
      if (dyn_cast<triton::LoadOp>(op) || dyn_cast<triton::StoreOp>(op) ||
          dyn_cast<triton::amdgpu::BufferLoadOp>(op) ||
          dyn_cast<triton::amdgpu::BufferStoreOp>(op)) {
        if (MachineModel::maxLoadStoreIssues > issuedLoadStoreCounter) {
          ++issuedLoadStoreCounter;
          result.set(node);
        } else {
          result.lowPriorityNodes.push_back(node);
        }
        continue;
      }
      if (dyn_cast<triton::DotOp>(op)) {
        issuedLocalStoreLoadCounter =
            std::max(0, issuedLocalStoreLoadCounter - 1);
        issuedLoadStoreCounter = std::max(0, issuedLoadStoreCounter - 1);
        result.set(node);
        continue;
      }
      if (dyn_cast<mlir::gpu::BarrierOp>(op)) {
        result.set(node);
        continue;
      }
      result.normPriorityNodes.push_back(node);
    }

    return result;
  }

  void printState(llvm::raw_ostream &stream) {
    stream << "issuedLoadStoreCounter: " << issuedLoadStoreCounter << "; "
           << "issuedLocalStoreLoadCounter: " << issuedLocalStoreLoadCounter
           << '\n';
  }

private:
  const inline static int32_t maxLoadStoreIssues{2};
  const inline static int32_t maxLocalLoadStoreIssues{6};
  int32_t issuedLoadStoreCounter{0};
  int32_t issuedLocalStoreLoadCounter{0};
};

struct TritonAMDGPURescheduleOps
    : public TritonAMDGPURescheduleOpsBase<TritonAMDGPURescheduleOps> {
  explicit TritonAMDGPURescheduleOps(StringRef targetArch) {
    this->arch = targetArch.str();
  }

  LogicalResult verify(Block *mlirBlock) {
    // make sure that a block gets terminated with `cf::BranchOp`
    if (!dyn_cast<cf::BranchOp>(&(mlirBlock->back()))) {
      return failure();
    }

    // do't schedule if there is not enough operations in a block
    if (mlirBlock->getOperations().size() < 3)
      return failure();
    return success();
  }

  void reschedule(Block *mlirBlock) {
    Graph graph(mlirBlock);

    GraphManager manager(graph);
    MachineModel machineModel;
    SmallVector<Operation *> rescheduledOps;

    auto defaultSelector = [&](const SmallVector<Node *> readyNodes) {
      size_t minSourceCodeNodeIndex = std::numeric_limits<size_t>::max();
      Node *earliestNodeToRun = nullptr;
      for (auto node : readyNodes) {
        const auto sourceCodeIndex = manager.getNodeSourceCodeIndex(node);
        if (minSourceCodeNodeIndex > sourceCodeIndex) {
          minSourceCodeNodeIndex = sourceCodeIndex;
          earliestNodeToRun = node;
        }
      }
      return earliestNodeToRun;
    };

    bool verbose = false;
    std::string dbgStr;
    llvm::raw_string_ostream dbgStream(dbgStr);
    while (!manager.finished()) {
      const auto &readyNodes = manager.getCurrentLeafs();
      MachineModel::Result selectionResult = machineModel.select(readyNodes);
      auto selectedNode = selectionResult.selectedNode;
      bool selectedFromMachineModel = selectedNode ? true : false;

      bool selectedFromNormPrioQueue = false;
      if (!selectedNode) {
        selectedNode = defaultSelector(selectionResult.normPriorityNodes);
        selectedFromNormPrioQueue = true;
      }

      bool selectedFromLowPrioqueue = false;
      if (!selectedNode) {
        selectedNode = defaultSelector(selectionResult.lowPriorityNodes);
        selectedFromLowPrioqueue = true;
      }

      assert(selectedNode != nullptr);

      if (verbose) {
        dbgStream << std::string(80, '+') << "\n";
        for (auto n : selectionResult.normPriorityNodes) {
          n->getOp()->print(dbgStream);
          dbgStream << '\n';
        }
        dbgStream << "\n\n\nSelected\n";
        selectedNode->getOp()->print(dbgStream);
        dbgStream << '\n';
        machineModel.printState(dbgStream);
        dbgStream << "selectedFromMachineModel: " << selectedFromMachineModel
                  << "; "
                  << "selectedFromNormPrioQueue: " << selectedFromNormPrioQueue
                  << "; "
                  << "selectedFromLowPrioqueue: " << selectedFromLowPrioqueue
                  << '\n';
      }

      manager.removeLeaf(selectedNode);
      rescheduledOps.push_back(selectedNode->getOp());
    }

    if (verbose)
      llvm::outs() << dbgStream.str() << '\n';

    std::string outStr;
    llvm::raw_string_ostream outStream(outStr);
    outStream << "\n\n\n...." << std::string(80, '-') << '\n';
    for (auto op : rescheduledOps) {
      op->print(outStream);
      outStream << "\n";
    }

    // TODO: put `rescheduledOps` to the MLIR code
    // (either create a new MLIR block or insert instruction into the existing
    // one)
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    llvm::SmallVector<Block *> blocks;
    mod.walk([&](amdgpu::InstructionSchedHint hint) {
      if (hint.getVariant() == amdgpu::SchedHint::refine_ops) {
        blocks.push_back(hint->getBlock());
        hint->erase();
      }
    });

    for (auto block : blocks) {
      if (succeeded(verify(block)))
        reschedule(block);
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
