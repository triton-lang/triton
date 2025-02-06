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

  template <ChildType Type> void add(Node *node) {
    llvm::SetVector<Node *> *children =
        Type == ChildType::Real ? &realChildren : &artificialChildren;
    children->insert(node);
  }

  size_t getNumParents() { return parents.size(); }
  void addParent(Node *node) { parents.insert(node); }

  void removeParent(Node *node) {
    if (parents.contains(node)) {
      parents.remove(node);
    }
  }

  void drainChildren() {
    realChildren.clear();
    artificialChildren.clear();
  }

  bool empty() { return realChildren.empty() && artificialChildren.empty(); }

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

  llvm::SmallVector<std::unique_ptr<Node>> &getNodes() { return nodes; }

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
      auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(node->op);
      auto localStore = llvm::dyn_cast<triton::gpu::LocalStoreOp>(node->op);
      if (localLoad || localStore) {
        ldsOpsNodes.push_back(node);
      }
      auto gpuBarrier = llvm::dyn_cast<mlir::gpu::BarrierOp>(node->op);
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
      for (auto operandValue : node->op->getOperands()) {
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
    std::string name = std::to_string(reinterpret_cast<intptr_t>(node.get()));
    out << name << "\t[label=\"" << node->op->getName() << "\"]\n";
  }
  for (auto [idx, node] : llvm::enumerate(graph.getNodes())) {
    std::string name = std::to_string(reinterpret_cast<intptr_t>(node.get()));
    for (auto child : node->realChildren) {
      std::string childName = std::to_string(reinterpret_cast<intptr_t>(child));
      out << "\t" << childName << " -> " << name << ";\n";
    }
    for (auto child : node->artificialChildren) {
      std::string childName = std::to_string(reinterpret_cast<intptr_t>(child));
      out << "\t" << childName << " -> " << name
          << " [style=\"dashed\", color=\"blue\"];\n";
    }
  }
  out << "}";
  return out;
}

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
    llvm::outs() << graph << '\n';

    // TODO: build dependency graph
    // TODO: use gpu.barrier as havy-edges
    // TODO: move ops around to improve ILP
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
