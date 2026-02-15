#ifndef TRITON_TRITONGPU_TRANSFORMS_PARTITION_SCHEDULING_UTILITY_H_
#define TRITON_TRITONGPU_TRANSFORMS_PARTITION_SCHEDULING_UTILITY_H_

#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton::gpu::partition_scheduling_detail {

namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

class Graph;
class Node;

enum Flags : uint8_t {
  NONE = 0,
  MANUAL = 1 << 0,
  LOAD = 1 << 1,
  STORE = 1 << 2,
  MMA = 1 << 3,
  TMEM = 1 << 4,
  SFU = 1 << 5,
  VIEW = 1 << 6,
};

inline Flags &operator|=(Flags &lhs, Flags rhs) {
  return lhs = static_cast<Flags>(lhs | rhs);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &stream, Flags flags);

Flags getNodeFlags(Node *node);

size_t computeCost(Operation *op);

inline bool isViewOp(Operation *op) {
  return isa<tt::BroadcastOp, tt::ExpandDimsOp, ttg::ConvertLayoutOp>(op) ||
         op->hasTrait<OpTrait::MemDescViewTrait>();
}

class Partition {
public:
  explicit Partition(Graph *graph) : graph(graph) {}
  void add(Node *node);
  void remove(Node *node) { nodes.remove(node); }
  void addFlag(Flags flag) { flags |= flag; }
  Flags getFlags() const { return flags; }
  const SetVector<Node *> &getNodes() const { return nodes; }
  bool empty() const { return nodes.empty(); }

  size_t getStage() const {
    if (flags & Flags::MMA)
      return 1;
    return 0;
  }
  size_t getCost() const { return cost; }

  static void merge(Partition *lhs, Partition *rhs);

  void dump() const;

  std::optional<size_t> id;

private:
  Graph *graph;
  Flags flags = Flags::NONE;
  size_t cost = 0;
  SetVector<Node *> nodes;
};

class Port {
public:
  Port() = default;
  Port(Node *node, size_t idx) : node(node), idx(idx) {}
  Node *getNode() const { return node; }
  size_t getIdx() const { return idx; }

  bool operator==(const Port &other) const {
    return node == other.node && idx == other.idx;
  }

private:
  Node *node = nullptr;
  size_t idx = 0;
};

} // namespace mlir::triton::gpu::partition_scheduling_detail

namespace llvm {
template <>
struct DenseMapInfo<mlir::triton::gpu::partition_scheduling_detail::Port> {
  static inline mlir::triton::gpu::partition_scheduling_detail::Port
  getEmptyKey() {
    return {};
  }

  static inline mlir::triton::gpu::partition_scheduling_detail::Port
  getTombstoneKey() {
    return mlir::triton::gpu::partition_scheduling_detail::Port(0, 1);
  }

  static unsigned getHashValue(
      const mlir::triton::gpu::partition_scheduling_detail::Port &port) {
    return std::hash<mlir::triton::gpu::partition_scheduling_detail::Node *>()(
               port.getNode()) ^
           std::hash<size_t>()(port.getIdx());
  }

  static bool
  isEqual(const mlir::triton::gpu::partition_scheduling_detail::Port &lhs,
          const mlir::triton::gpu::partition_scheduling_detail::Port &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace mlir::triton::gpu::partition_scheduling_detail {

using InputPort = Port;
using OutputPort = Port;

class Edge {
public:
  Edge() = default;
  Edge(OutputPort from, InputPort to) : from(from), to(to) {}

  OutputPort getFrom() const { return from; }
  InputPort getTo() const { return to; }

  Node *getFromNode() const { return from.getNode(); }
  size_t getFromIdx() const { return from.getIdx(); }

  Node *getToNode() const { return to.getNode(); }
  size_t getToIdx() const { return to.getIdx(); }

  bool isDataValue() const;
  bool crossesPartitions() const;
  Type getType() const;
  size_t getSize() const;

private:
  OutputPort from;
  InputPort to;
};

class Node {
public:
  explicit Node(Operation *op) : op(op), cost(computeCost(op)) {}

  Node(Node *parent, Operation *op, size_t numInputs, size_t numOutputs)
      : parent(parent), op(op), cost(computeCost(op)) {
    inputs.resize(numInputs);
    outputs.resize(numOutputs);
    dataOutputs.resize(numOutputs);
  }

  Node(Node *parent, Value value, size_t numInputs, size_t numOutputs)
      : parent(parent), value(value) {
    inputs.resize(numInputs);
    outputs.resize(numOutputs);
    dataOutputs.resize(numOutputs);
  }

  Node *addNode(Operation *op, size_t inputs, size_t outputs) {
    return nodes.emplace_back(new Node(this, op, inputs, outputs)).get();
  }

  Node *addNode(Value value, size_t inputs, size_t outputs) {
    return nodes.emplace_back(new Node(this, value, inputs, outputs)).get();
  }

  void walk(const std::function<void(Node *)> &fn) {
    std::function<void(Node *)> do_walk = [&](Node *node) {
      for (auto &child : node->getNodes()) {
        fn(child.get());
        do_walk(child.get());
      }
    };
    do_walk(this);
  }

  static void addEdge(OutputPort from, InputPort to) {
    from.getNode()->addOutputEdge(from.getIdx(), to);
    to.getNode()->addInputEdge(to.getIdx(), from);
  }

  static void removeEdge(Edge edge) {
    edge.getFromNode()->removeOutputEdge(edge.getFromIdx(), edge.getTo());
    edge.getToNode()->removeInputEdge(edge.getToIdx(), edge.getFrom());
  }

  void addDefines(Node *node) { defines.push_back(node); }

  void addInputEdge(size_t idx, OutputPort port) {
    assert(idx < inputs.size());
    inputs[idx] = port;
  }

  void removeInputEdge(size_t idx, OutputPort port) {
    assert(idx < inputs.size());
    inputs[idx] = {};
  }

  void addOutputEdge(size_t idx, InputPort port) {
    assert(idx < outputs.size());
    outputs[idx].push_back(port);
  }

  void removeOutputEdge(size_t idx, InputPort port) {
    assert(idx < outputs.size());
    for (auto it = outputs[idx].begin(); it != outputs[idx].end(); it++) {
      if (*it == port) {
        outputs[idx].erase(it);
        break;
      }
    }
  }

  Node *getParent() const { return parent; }
  bool isOp() const { return op; }
  bool isValue() const { return !op; }
  Operation *getOp() { return op; }
  Value &getValue() {
    assert(isValue());
    return value;
  }
  const SmallVector<Node *> &getDefines() const { return defines; }

  const SmallVector<std::unique_ptr<Node>> &getNodes() const { return nodes; }

  size_t getNumInputs() const { return inputs.size(); }
  size_t getNumOutputs() const { return outputs.size(); }

  const SmallVector<OutputPort> &getInputs() const { return inputs; }
  const SmallVector<SmallVector<InputPort>> &getOutputs() const {
    return outputs;
  }
  SmallVector<InputPort> getOutputsFromPort(size_t idx) const {
    return outputs[idx];
  }

  SmallVector<Edge> getInEdges() {
    SmallVector<Edge> result;
    size_t idx = 0;
    for (auto input : inputs) {
      result.push_back(Edge(input, InputPort(this, idx)));
      idx++;
    }
    return result;
  }

  size_t getNumInDataEdges() {
    size_t count = 0;
    size_t idx = 0;
    for (auto input : inputs) {
      Edge edge(input, InputPort(this, idx));
      if (edge.isDataValue())
        count++;
      idx++;
    }
    return count;
  }

  SmallVector<Edge> getOutEdges() {
    SmallVector<Edge> result;
    size_t idx = 0;
    for (auto outputs : this->outputs) {
      for (auto output : outputs)
        result.push_back(Edge(OutputPort(this, idx), output));
      idx++;
    }
    return result;
  }

  size_t getNumOutDataEdges() {
    size_t count = 0;
    size_t idx = 0;
    for (auto output : dataOutputs) {
      if (output)
        count += outputs[idx].size();
      idx++;
    }
    return count;
  }

  void setDataValue(size_t idx) {
    assert(idx < dataOutputs.size());
    dataOutputs[idx] = true;
  }

  bool isDataValue(size_t idx) {
    assert(idx < dataOutputs.size());
    return dataOutputs[idx];
  }

  bool isData() {
    // node is data if it consumes/produces a data value
    if (std::any_of(dataOutputs.begin(), dataOutputs.end(),
                    [](bool x) { return x; })) {
      return true;
    }
    for (auto input : inputs)
      if (input.getNode() && input.getNode()->isDataValue(input.getIdx()))
        return true;
    return false;
  }

  bool containsData() {
    // node contains data if a data op appears in its region
    for (auto &node : getNodes()) {
      if (node->isData())
        return true;
      if (node->containsData())
        return true;
    }
    return false;
  }

  bool inLoopBody() {
    if (op)
      return op->getParentOfType<scf::ForOp>();
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      return isa<scf::ForOp>(parentOp) ||
             parentOp->getParentOfType<scf::ForOp>();
    }
    auto result = cast<OpResult>(value);
    auto op = result.getOwner();
    return isa<scf::ForOp>(op) || op->getParentOfType<scf::ForOp>();
  }

  bool containsLoopBody() {
    for (auto &node : getNodes()) {
      if (node->inLoopBody())
        return true;
      if (node->containsLoopBody())
        return true;
    }
    return false;
  }

  std::string getLabel() {
    if (op)
      return op->getName().getStringRef().str();
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (isa<tt::FuncOp>(parentOp))
        return "arg " + std::to_string(blockArg.getArgNumber());
      if (isa<scf::ForOp>(parentOp)) {
        if (blockArg.getArgNumber() == 0)
          return "ind var";
        return "iter arg " + std::to_string(blockArg.getArgNumber() - 1);
      }
      return "?";
    }
    auto result = cast<OpResult>(value);
    return "result " + std::to_string(result.getResultNumber());
  }

  void setPartition(Partition *partition) {
    for (auto current_partition : partitions)
      current_partition->remove(this);
    partitions.clear();
    partitions.insert(partition);
    partition->add(this);
  }

  void addPartition(Partition *partition) {
    partitions.insert(partition);
    partition->add(this);
  }

  void addPartitions(const SetVector<Partition *> &partitions) {
    this->partitions.insert(partitions.begin(), partitions.end());
    for (auto partition : partitions)
      partition->add(this);
  }

  bool hasPartition() const { return !partitions.empty(); }

  Partition *getPartition() const {
    assert(partitions.size() == 1);
    return *(partitions.begin());
  }

  const SetVector<Partition *> &getPartitions() const { return partitions; }

  bool hasCost() const { return cost > 0; }
  size_t getCost() const {
    assert(hasCost());
    return cost;
  }

  void dump() { llvm::errs() << "node '" << getLabel() << "'\n"; }

private:
  Node *parent = nullptr;
  Operation *op = nullptr;
  Value value;
  size_t cost = 0;

  SmallVector<std::unique_ptr<Node>> nodes;
  SmallVector<Node *> defines;

  SmallVector<OutputPort> inputs;
  SmallVector<SmallVector<InputPort>> outputs;
  SmallVector<bool> dataOutputs;

  SetVector<Partition *> partitions;
};

class Graph {
public:
  explicit Graph(Operation *op) : root(new Node(op)) {}

  Node *getRoot() { return root.get(); }

  Partition *addPartition() {
    auto partition = partition_storage.emplace_back(new Partition(this)).get();
    partitions.insert(partition);
    return partition;
  }

  void erasePartition(Partition *partition) {
    assert(partition->empty());
    partitions.remove(partition);
  }

  const SetVector<Partition *> &getPartitions() const { return partitions; }

  void walk(const std::function<void(Node *)> &fn) {
    std::function<void(Node *)> do_walk = [&](Node *node) {
      for (auto &child : node->getNodes()) {
        fn(child.get());
        do_walk(child.get());
      }
    };
    do_walk(root.get());
  }

private:
  std::unique_ptr<Node> root;
  SetVector<Partition *> partitions;
  SmallVector<std::unique_ptr<Partition>> partition_storage;
};

struct VisualizationInfo {
  DenseMap<Partition *, size_t> partition_ids;
  DenseMap<Partition *, std::string> partition_colors;
};

void visualize(std::string key, std::string filename, std::string title,
               Graph *graph, VisualizationInfo &info);

} // namespace mlir::triton::gpu::partition_scheduling_detail

#endif // TRITON_TRITONGPU_TRANSFORMS_PARTITION_SCHEDULING_UTILITY_H_
