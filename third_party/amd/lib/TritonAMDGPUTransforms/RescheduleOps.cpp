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

  Operation *op;
  llvm::SetVector<Node *> realChildren;
  llvm::SetVector<Node *> artificialChildren;
};

struct Graph {
public:
  Graph(Block *mlirBlock) {
    createNodes(mlirBlock);
    createEdges();
  }

  llvm::SmallVector<std::unique_ptr<Node>> &getNodes() { return nodes; }
  SmallVector<SetVector<Node *>> &getBlocks() { return blocks; }

private:
  SetVector<Node *> *getBlock(Node *node) {
    for (auto &block : blocks) {
      if (block.contains(node))
        return &block;
    }
    return nullptr;
  }

  void createNodes(Block *mlirBlock) {
    SetVector<Node *> currBlock;
    for (auto it = mlirBlock->begin(); it != mlirBlock->end(); ++it) {
      Operation *op = &(*it);
      std::unique_ptr<Node> node = std::make_unique<Node>(op);
      currBlock.insert(node.get());
      lookup.insert({op, node.get()});
      nodes.push_back(std::move(node));

      if (auto barrier = llvm::dyn_cast<mlir::gpu::BarrierOp>(op)) {
        if (!currBlock.empty()) {
          blocks.push_back(std::move(currBlock));
          currBlock = SetVector<Node *>{};
        }
      }
    }
    if (!currBlock.empty())
      blocks.push_back(std::move(currBlock));
  }

  void createEdges() {
    for (auto blockIt = blocks.rbegin(); blockIt != blocks.rend(); ++blockIt) {
      auto &currBlock = *blockIt;
      for (auto nodeIt = currBlock.rbegin(); nodeIt != currBlock.rend();
           ++nodeIt) {
        auto node = *nodeIt;
        for (auto operand : node->op->getOperands()) {
          Operation *childOp = operand.getDefiningOp();
          if (!lookup.contains(childOp))
            continue;
          Node *childNode = lookup.find(childOp)->second;
          if (currBlock.contains(childNode)) {
            node->add<Node::ChildType::Real>(childNode);
          } else {
            auto *childBlock = getBlock(childNode);
            assert(childBlock != nullptr);
            auto *lastNodeInChildBlock = childBlock->back();
            lastNodeInChildBlock->add<Node::ChildType::Artificials>(childNode);
          }
        }
        if (node->op->getUsers().empty()) {
          auto *lastNodeInBlock = currBlock.back();
          if (lastNodeInBlock != node) {
            lastNodeInBlock->add<Node::ChildType::Artificials>(node);
          }
        }
      }
    }
  }

  llvm::SmallVector<std::unique_ptr<Node>> nodes;
  SmallVector<SetVector<Node *>> blocks;
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
      out << "\t" << childName << " -> " << name << " [style=\"dashed\"];\n";
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
