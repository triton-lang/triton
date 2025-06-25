#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/Debug.h"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUPARTITIONSCHEDULING
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritongpu-partition-scheduling"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton;

namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

struct Options {
  bool dump_dot = false;
  bool dump_loop_only = false;
  bool dump_data_only = false;
  bool manual = false;
};

Options &get_options() {
  // FIXME: hacky having this as a global variable
  static Options global_options;
  return global_options;
}

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

Flags &operator|=(Flags &lhs, Flags rhs) {
  return lhs = static_cast<Flags>(lhs | rhs);
}

std::ostream &operator<<(std::ostream &stream, Flags flags) {
  std::vector<std::string> strs;
  if (flags == Flags::NONE) {
    strs.push_back("NONE");
  } else {
    if (flags & Flags::MANUAL)
      strs.push_back("MANUAL");
    if (flags & Flags::LOAD)
      strs.push_back("LOAD");
    if (flags & Flags::STORE)
      strs.push_back("STORE");
    if (flags & Flags::MMA)
      strs.push_back("MMA");
    if (flags & Flags::TMEM)
      strs.push_back("TMEM");
    if (flags & Flags::SFU)
      strs.push_back("SFU");
    if (flags & Flags::VIEW)
      strs.push_back("VIEW");
  }
  for (size_t i = 0; i < strs.size(); i++) {
    if (i != 0)
      stream << "|";
    stream << strs[i];
  }
  return stream;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &stream, Flags flags) {
  std::vector<std::string> strs;
  if (flags == Flags::NONE) {
    strs.push_back("NONE");
  } else {
    if (flags & Flags::MANUAL)
      strs.push_back("MANUAL");
    if (flags & Flags::LOAD)
      strs.push_back("LOAD");
    if (flags & Flags::STORE)
      strs.push_back("STORE");
    if (flags & Flags::MMA)
      strs.push_back("MMA");
    if (flags & Flags::TMEM)
      strs.push_back("TMEM");
    if (flags & Flags::SFU)
      strs.push_back("SFU");
    if (flags & Flags::VIEW)
      strs.push_back("VIEW");
  }
  for (size_t i = 0; i < strs.size(); i++) {
    if (i != 0)
      stream << "|";
    stream << strs[i];
  }
  return stream;
}

size_t computeCost(Operation *op) {
  if (auto mma = dyn_cast<ttng::TCGen5MMAOp>(op)) {
    auto a = mma.getA();
    auto b = mma.getB();
    auto a_shape = a.getType().getShape();
    auto b_shape = b.getType().getShape();
    assert(a_shape.size() == 2);
    assert(b_shape.size() == 2);
    auto M = a_shape[0];
    auto N = b_shape[0];
    auto K = a_shape[1];
    auto cycles = M * N * K / 8192;
    return cycles;
  }

  if (isa<math::Exp2Op, ElementwiseInlineAsmOp>(op)) {
    int elementCount = 0;
    for (Type type : op->getResultTypes()) {
      if (auto tensorTy = dyn_cast<RankedTensorType>(type))
        elementCount += tensorTy.getNumElements();
    }
    return elementCount;
    // if (elementCount > 256) {
    //   setPartition(&op, defaultPartition);
    //   scheduleDependencies(loop, partitions, defaultPartition, &op);
    // }
  }

  // auto inp = exp2.getOperand();
  // auto shape = cast<TensorType>(inp.getType()).getShape();
  // size_t size = 1;
  // for (auto x : shape)
  //   size *= x;
  // auto cycles = size / 16;
  // return cycles;
  //}

  return 0;
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
    // FIXME: correct behaviour?
    if (flags & Flags::MMA)
      return 1;
    return 0;
  }
  size_t getCost() const { return cost; }

  static void merge(Partition *lhs, Partition *rhs);

  void dump() const {
    llvm::errs() << "Partition@" << this << " {\n"
                 << "  id=" << id << "\n"
                 << "  size=" << nodes.size() << "\n"
                 << "  cost=" << cost << "\n"
                 << "  flags=" << flags << "\n"
                 << "}\n";
  }

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

} // namespace
} // namespace mlir::triton::gpu
namespace llvm {
template <> struct DenseMapInfo<mlir::triton::gpu::Port> {
  static inline mlir::triton::gpu::Port getEmptyKey() { return {}; }

  static inline mlir::triton::gpu::Port getTombstoneKey() {
    return mlir::triton::gpu::Port(0, 1);
  }

  static unsigned getHashValue(const mlir::triton::gpu::Port &port) {
    return std::hash<mlir::triton::gpu::Node *>()(port.getNode()) ^
           std::hash<size_t>()(port.getIdx());
  }

  static bool isEqual(const mlir::triton::gpu::Port &lhs,
                      const mlir::triton::gpu::Port &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm
namespace mlir::triton::gpu {
namespace {

using InputPort = Port;
using OutputPort = Port;

class Edge {
public:
  Edge() = default;
  Edge(OutputPort from, InputPort to) : from(from), to(to) {}

  Node *getFromNode() const;
  size_t getFromIdx() const;
  Node *getToNode() const;
  size_t getToIdx() const;

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

  void addDefines(Node *node) { defines.push_back(node); }

  void addInputEdge(size_t idx, OutputPort port) {
    assert(idx < inputs.size());
    inputs[idx] = port;
  }

  void addOutputEdge(size_t idx, InputPort port) {
    assert(idx < outputs.size());
    outputs[idx].push_back(port);
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
    if (auto result = dyn_cast<OpResult>(value)) {
      auto op = result.getOwner();
      return isa<scf::ForOp>(op) || op->getParentOfType<scf::ForOp>();
    }
    llvm::report_fatal_error("unsuported value");
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
      llvm::report_fatal_error("unsuported op");
    }
    if (auto result = dyn_cast<OpResult>(value))
      return "result " + std::to_string(result.getResultNumber());
    llvm::report_fatal_error("unsuported value");
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
    return partitions.emplace_back(new Partition(this)).get();
  }

  const SmallVector<std::unique_ptr<Partition>> &getPartitions() const {
    return partitions;
  }

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
  SmallVector<std::unique_ptr<Partition>> partitions;
};

template <typename... Args> bool node_isa(Node *node) {
  return node->isOp() && isa<Args...>(node->getOp());
}

bool isScalarLoad(Node *node) {
  if (!node_isa<tt::LoadOp>(node))
    return false;

  auto op = cast<tt::LoadOp>(node->getOp());

  op->getResult(0).getType().dump();
  if (auto tensorType =
          dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType())) {
    for (auto dim : tensorType.getShape())
      if (dim != 1)
        return false;
    return true;
  }

  return true;
}

bool isAsyncLoad(Node *node) {
  // Special case:
  // tt.load that occurs in a sequence:
  //    tt.load -> ttg.local_alloc -> ttng.tc_gen5_mma
  // is placed in a load partition, and later lowered to cp.async
  if (node_isa<tt::LoadOp>(node)) {
    auto outs = node->getOutEdges();
    if (outs.size() == 1) {
      auto local_alloc = outs.front().getToNode();
      if (local_alloc->getOp() &&
          isa<ttg::LocalAllocOp>(local_alloc->getOp())) {
        auto outs = local_alloc->getOutEdges();
        if (outs.size() == 1) {
          auto mma = outs.front().getToNode();
          if (mma->getOp() && isa<ttng::MMAv5OpInterface>(mma->getOp()))
            return true;
        }
      }
    }
  }
  return false;
}

bool isViewOp(Operation *op) {
  return isa<tt::BroadcastOp, tt::ExpandDimsOp, ttg::ConvertLayoutOp>(op) ||
         op->hasTrait<OpTrait::MemDescViewTrait>();
}

Flags getNodeFlags(Node *node) {
  if (node->isOp()) {
    auto op = node->getOp();

    // if it is manually tagged with a node type
    if (op->hasAttr("store"))
      return Flags::STORE;

    if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op)
        //|| (isAsyncLoad(node) && !isScalarLoad(node))
    )
      return Flags::LOAD;
    if (isa<tt::DescriptorStoreOp, tt::DescriptorScatterOp>(op))
      return Flags::STORE;
    if (isa<ttng::MMAv5OpInterface>(op))
      return Flags::MMA;
    if (isa</*ttng::TMEMAllocOp,*/ ttng::TMEMLoadOp, ttng::TMEMStoreOp>(op))
      return Flags::TMEM;
    if (isa<math::Exp2Op>(op))
      return Flags::SFU;
    if (isViewOp(op))
      return Flags::VIEW;
  }
  return Flags::NONE;
}

void Partition::add(Node *node) {
  auto node_flags = getNodeFlags(node);

  // Note: only set view flag for partition,
  // if it consists of all view ops
  // FIXME: have a set kinds of flag to make this generic?
  bool all_view = true;
  if (!nodes.empty() && !(flags & Flags::VIEW))
    all_view = false;
  if (!(node_flags & Flags::VIEW))
    all_view = false;

  nodes.insert(node);

  flags |= node_flags;
  if (!all_view)
    flags = static_cast<Flags>(flags & ~Flags::VIEW);

  if (node->hasCost())
    cost += node->getCost();
}

void Partition::merge(Partition *lhs, Partition *rhs) {
  // Should never be merging MANUAL partitions
  assert(!((lhs->getFlags() & Flags::MANUAL) &&
           (rhs->getFlags() & Flags::MANUAL)));

  // Always keep the MANUAL partition,
  // and prefer emptying the NONE partition
  if (lhs->getFlags() & Flags::MANUAL || rhs->getFlags() == Flags::NONE)
    std::swap(lhs, rhs);

  auto nodes = lhs->getNodes();
  for (auto node : nodes) {
    node->setPartition(rhs);
  }
  // FIXME: remove empty partitions? we just ignore them in later parts of the
  // code
}

Node *Edge::getFromNode() const { return from.getNode(); }
size_t Edge::getFromIdx() const { return from.getIdx(); }

Node *Edge::getToNode() const { return to.getNode(); }
size_t Edge::getToIdx() const { return to.getIdx(); }

bool Edge::isDataValue() const {
  if (!from.getNode())
    return false;
  return from.getNode()->isDataValue(from.getIdx());
}

bool Edge::crossesPartitions() const {
  if (!isDataValue())
    return false;
  if (!from.getNode()->hasPartition() || !to.getNode()->hasPartition())
    return false;
  // FIXME: may not handle multiple partitions correctly
  if (from.getNode()->getPartitions().size() != 1 ||
      to.getNode()->getPartitions().size() != 1)
    return false;
  return from.getNode()->getPartition() != to.getNode()->getPartition();
}

Type Edge::getType() const {
  auto fromNode = from.getNode();
  if (fromNode->isOp())
    return fromNode->getOp()->getResult(from.getIdx()).getType();
  return fromNode->getValue().getType();
}

size_t Edge::getSize() const {
  // TODO: do we want to include the element type? i.e. return the number of
  // bytes?
  auto type = getType();

  if (auto tensor = dyn_cast<TensorType>(type)) {
    size_t size = 1;
    for (auto x : tensor.getShape())
      size *= x;
    return size;
  }

  if (auto memdesc = dyn_cast<MemDescType>(type)) {
    size_t size = 1;
    for (auto x : memdesc.getShape())
      size *= x;
    return size;
  }

  return 1;
}

struct VisualizationInfo {
  DenseMap<Partition *, size_t> partition_ids;
  DenseMap<Partition *, std::string> partition_colors;
};

void visualize(std::string key, std::string filename, std::string title,
               Graph *graph, VisualizationInfo &info);

std::unique_ptr<Graph> buildGraph(Operation *region) {
  DenseMap<Operation *, Node *> nodes;
  DenseMap<std::pair<Operation *, size_t>, InputPort> operands;
  SmallVector<std::pair<OutputPort, Value>> values;

  std::function<void(Node * graph, Operation *)> visitOperation =
      [&](Node *graph, Operation *op) {
        if (auto funcOp = dyn_cast<FuncOp>(op)) {
          auto node = graph->addNode(op, 0, 0);
          nodes[op] = node;
          for (size_t idx = 0; idx < funcOp.getNumArguments(); idx++) {
            auto argNode = node->addNode(funcOp.getArgument(idx), 0, 1);
            values.push_back(std::make_pair(OutputPort(argNode, 0),
                                            funcOp.getArgument(idx)));
          }
          for (auto &region : op->getRegions())
            for (auto &block : region)
              for (auto &op : block)
                visitOperation(node, &op);

        } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
          auto node = graph->addNode(op, 3, 0);
          nodes[op] = node;

          // lb / ub / step
          operands[std::make_pair(op, 0)] = InputPort(node, 0);
          operands[std::make_pair(op, 1)] = InputPort(node, 1);
          operands[std::make_pair(op, 2)] = InputPort(node, 2);

          // iter args / results
          auto ind_var = node->addNode(forOp.getInductionVar(), 0, 1);
          node->addDefines(ind_var);
          values.push_back(
              std::make_pair(OutputPort(ind_var, 0), forOp.getInductionVar()));
          size_t idx = 0;
          for (auto iter_arg : forOp.getRegionIterArgs()) {
            auto iter_arg_node = node->addNode(iter_arg, 2, 1);
            node->addDefines(iter_arg_node);
            values.push_back(
                std::make_pair(OutputPort(iter_arg_node, 0), iter_arg));
            values.push_back(std::make_pair(OutputPort(iter_arg_node, 0),
                                            forOp.getResult(idx)));
            idx++;
          }

          // init iter args
          {
            size_t idx = 0;
            for (auto operand : forOp.getInitArgs()) {
              auto iter_arg_node = node->getDefines()[idx + 1];
              operands[std::make_pair(op, idx + 3)] =
                  InputPort(iter_arg_node, 0);
              idx++;
            }
          }

          for (auto &region : op->getRegions())
            for (auto &block : region)
              for (auto &op : block)
                visitOperation(node, &op);

        } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          auto node = graph->addNode(op, 1, 0);
          nodes[op] = node;

          // cond
          operands[std::make_pair(op, 0)] = InputPort(node, 0);

          // results
          for (auto result : ifOp.getResults()) {
            auto result_node = node->addNode(result, 2, 1);
            node->addDefines(result_node);
            values.push_back(
                std::make_pair(OutputPort(result_node, 0), result));
          }

          for (auto &region : op->getRegions())
            for (auto &block : region)
              for (auto &op : block)
                visitOperation(node, &op);

        } else if (auto reduceOp = dyn_cast<tt::ReduceOp>(op)) {

          auto node = graph->addNode(op, 1, 1);
          nodes[op] = node;

          // input
          operands[std::make_pair(op, 0)] = InputPort(node, 0);

          // result
          assert(reduceOp.getResults().size() == 1);
          auto result = reduceOp.getResults().front();
          values.push_back(std::make_pair(OutputPort(node, 0), result));

          for (auto &region : op->getRegions())
            for (auto &block : region)
              for (auto &op : block)
                visitOperation(node, &op);

        } else if (isa<scf::YieldOp>(op)) {

          if (auto forOp = dyn_cast<scf::ForOp>(op->getParentOp())) {
            // map operands to yield in a for op to the iter arg nodes
            auto for_node = nodes[op->getParentOp()];
            for (size_t idx = 0; idx < op->getNumOperands(); idx++) {
              auto block_arg_node =
                  for_node->getDefines()[idx + 1]; // skip iter arg
              operands[std::make_pair(op, idx)] = InputPort(block_arg_node, 1);
            }

          } else if (auto ifOp = dyn_cast<scf::IfOp>(op->getParentOp())) {
            // map operands to yield in an if op to the if results
            auto if_node = nodes[op->getParentOp()];
            for (size_t idx = 0; idx < op->getNumOperands(); idx++) {
              auto result_node = if_node->getDefines()[idx];
              operands[std::make_pair(op, idx)] = InputPort(
                  result_node,
                  (op->getParentRegion() == &ifOp.getThenRegion()) ? 0 : 1);
            }
          } else {
            assert(false && "unsupported");
          }

        } else if (isa<tt::ReturnOp>(op)) {
          // omit

        } else {
          // assert(op->getNumRegions() == 0);
          auto node =
              graph->addNode(op, op->getNumOperands(), op->getNumResults());
          nodes[op] = node;
          for (size_t idx = 0; idx < op->getNumOperands(); idx++)
            operands[std::make_pair(op, idx)] = InputPort(node, idx);
          for (const auto &result : op->getResults())
            values.push_back(std::make_pair(
                OutputPort(node, result.getResultNumber()), result));
        }
      };

  auto graph = std::make_unique<Graph>(region);
  if (auto m = dyn_cast<ModuleOp>(region)) {
    // Special case for analyzing entire module
    for (auto &block : m.getRegion())
      for (auto &op : block)
        visitOperation(graph->getRoot(), &op);
  } else {
    visitOperation(graph->getRoot(), region);
  }

  for (auto [outputPort, value] : values) {
    for (auto &use : value.getUses()) {
      auto op = use.getOwner();
      if (op) {
        auto key = std::make_pair(op, use.getOperandNumber());
        if (operands.find(key) != operands.end()) {
          auto inputPort = operands[key];
          Node::addEdge(outputPort, inputPort);
        } else {
          // llvm::report_fatal_error(
          //    "use not found for op when constructing data flow graph");
        }
      } else {
        llvm::report_fatal_error(
            "use not owned by an op when constructing data flow graph");
      }
    }
  }

  return graph;
}

SmallVector<OutputPort> initialDataValues(Graph *graph) {
  SmallVector<OutputPort> values;
  graph->walk([&](Node *node) {
    if (node->isOp()) {
      auto op = node->getOp();
      if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op)) {
        node->setDataValue(0);
        values.push_back({node, 0});
      }
      // if (isa<tt::LoadOp>(op)) {
      //   if (isAsyncLoad(node) && !isScalarLoad(node)) {
      //     node->setDataValue(0);
      //     values.push_back({node, 0});
      //   }
      // }
      if (isa<ttng::TMEMLoadOp>(op)) {
        node->setDataValue(0);
        values.push_back({node, 0});
        node->setDataValue(1);
        values.push_back({node, 1});
      }
      // if it is manually tagged with data attribute,
      // all outputs are treated as data values
      if (op->hasAttr("data")) {
        for (size_t i = 0; i < node->getNumOutputs(); i++) {
          node->setDataValue(i);
          values.push_back({node, i});
        }
      }
    }
  });
  return values;
}

void propagateDataValues(const SmallVector<OutputPort> &values) {
  SmallVector<OutputPort> stack = values;
  DenseSet<OutputPort> seen;
  seen.insert(values.begin(), values.end());

  auto add = [&](OutputPort value) {
    value.getNode()->setDataValue(value.getIdx());
    if (seen.find(value) == seen.end()) {
      stack.push_back(value);
      seen.insert(value);
    }
  };

  while (!stack.empty()) {
    auto value = stack.back();
    stack.pop_back();
    for (auto use : value.getNode()->getOutputsFromPort(value.getIdx())) {
      auto use_node = use.getNode();
      for (size_t idx = 0; idx < use_node->getNumOutputs(); idx++) {
        OutputPort new_value{use_node, idx};
        add(new_value);
      }
    }
  }
}

void initialPartitionAssignment(Graph *graph) {
  graph->walk([&](Node *node) {
    if (node->isData() && !node->hasPartition()) {
      auto partition = graph->addPartition();
      node->setPartition(partition);
    }
  });
}

SmallVector<Edge> getCrossingEdges(Graph *graph) {
  SmallVector<Edge> edges;
  for (auto &partition : graph->getPartitions())
    for (auto node : partition->getNodes())
      for (auto edge : node->getOutEdges()) {
        if (!edge.crossesPartitions())
          continue;
        edges.push_back(edge);
      }
  return edges;
}

SmallVector<Edge> getInCrossingEdges(Partition *partition) {
  SmallVector<Edge> edges;
  for (auto node : partition->getNodes())
    for (auto edge : node->getInEdges()) {
      if (!edge.crossesPartitions())
        continue;
      edges.push_back(edge);
    }
  return edges;
}

SmallVector<Edge> getOutCrossingEdges(Partition *partition) {
  SmallVector<Edge> edges;
  for (auto node : partition->getNodes())
    for (auto edge : node->getOutEdges()) {
      if (!edge.crossesPartitions())
        continue;
      edges.push_back(edge);
    }
  return edges;
}

bool deserializeManualPartitions(Operation *region, Graph *graph) {
  std::map<int, Partition *> manual_partitions;
  graph->walk([&](Node *node) {
    if (node->isOp()) {
      auto op = node->getOp();
      if (op->hasAttr(kPartitionAttrName)) {
        auto partitionIds =
            cast<DenseI32ArrayAttr>(op->getAttr(kPartitionAttrName))
                .asArrayRef();
        for (auto id : partitionIds) {
          if (manual_partitions.find(id) == manual_partitions.end()) {
            auto partition = graph->addPartition();
            partition->addFlag(Flags::MANUAL);
            manual_partitions[id] = partition;

            LLVM_DEBUG({
              llvm::errs() << "deserialize manual partition:";
              partition->dump();
            });
          }
          node->addPartition(manual_partitions[id]);
        }
      }
    }
  });

  return !manual_partitions.empty();
}

bool isNone(Node *node) {
  auto partition = node->getPartition();
  auto flags = partition->getFlags();
  return flags == Flags::NONE || flags == Flags::MANUAL;
}

bool isOnlyNone(Node *node) {
  auto partition = node->getPartition();
  auto flags = partition->getFlags();
  return flags == Flags::NONE;
}

bool isView(Node *node) {
  auto partition = node->getPartition();
  auto flags = partition->getFlags();
  return flags & Flags::VIEW;
}

bool isManual(Node *node) {
  auto partition = node->getPartition();
  auto flags = partition->getFlags();
  return flags & Flags::MANUAL;
}

bool isLoad(Node *node) {
  auto partition = node->getPartition();
  auto flags = partition->getFlags();
  return flags & Flags::LOAD;
}

bool isStore(Node *node) {
  auto partition = node->getPartition();
  auto flags = partition->getFlags();
  return flags & Flags::STORE;
}

bool isMMA(Node *node) {
  auto partition = node->getPartition();
  auto flags = partition->getFlags();
  return flags & Flags::MMA;
}

bool isTMEM(Node *node) {
  auto partition = node->getPartition();
  auto flags = partition->getFlags();
  return flags & Flags::TMEM;
}

bool isSFU(Node *node) {
  auto partition = node->getPartition();
  auto flags = partition->getFlags();
  return flags & Flags::SFU;
}

bool isCostlySFU(Node *node) {
  auto partition = node->getPartition();
  auto flags = partition->getFlags();
  return (flags & Flags::SFU) && partition->getCost() > 256;
}

bool isForIterArg(Node *node) {
  if (node->isOp())
    return false;
  auto blockArg = dyn_cast<BlockArgument>(node->getValue());
  if (!blockArg)
    return false;
  return isa<scf::ForOp>(blockArg.getOwner()->getParentOp());
}

bool isIfResult(Node *node) {
  if (node->isOp())
    return false;
  auto result = dyn_cast<OpResult>(node->getValue());
  if (!result)
    return false;
  return isa<scf::IfOp>(result.getOwner());
}

SmallVector<std::pair<std::string, std::function<bool(Edge)>>> heuristics = {
    // load followed by local alloc in same partition
    {"load_local_alloc",
     [](Edge edge) {
       if (!node_isa<ttg::LocalAllocOp>(edge.getToNode())) {
         return false;
       }

       if (node_isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(
               edge.getFromNode())) {
         // require layouts to match for TMA load + alloc
         auto load = edge.getFromNode()->getOp();
         auto alloc = cast<ttg::LocalAllocOp>(edge.getToNode()->getOp());
         return getSharedEncoding(load) == alloc.getType().getEncoding();
       }

       if (node_isa<tt::LoadOp>(edge.getFromNode())) {
         return true;
       }

       return false;
     }},

    // sequence of view ops in same partition
    // Note: view ops guaranteed to have been duplicated so there
    // is one use/def for each
    {"view_sequence",
     [](Edge edge) {
       auto from = getNodeFlags(edge.getFromNode());
       auto to = getNodeFlags(edge.getToNode());
       return (from & Flags::VIEW) && (to & Flags::VIEW);
     }},

    // merge view op partition with producer if it involves fewer
    // elements than merging with the consumer of the view partition
    {"view_producer",
     [](Edge edge) {
       if (!isView(edge.getToNode())) {
         return false;
       }
       auto from = getNodeFlags(edge.getFromNode());
       auto to = getNodeFlags(edge.getToNode());
       if (!(to & Flags::VIEW)) {
         return false;
       }

       auto view_partition = edge.getToNode()->getPartition();
       auto out_edges = getOutCrossingEdges(view_partition);
       if (out_edges.size() != 1) {
         assert(false); // FIXME: should never reach here?
         return false;
       }
       auto out_edge = out_edges[0];

       auto in_size = edge.getSize();
       auto out_size = out_edge.getSize();

       return in_size > out_size;
     }},

    // merge remaining view op partitions with consumer
    // as that involves fewer elements being communicated via aref
    {"view_consumer",
     [](Edge edge) {
       if (!isView(edge.getFromNode())) {
         return false;
       }
       auto from = getNodeFlags(edge.getFromNode());
       auto to = getNodeFlags(edge.getToNode());
       if (!(from & Flags::VIEW)) {
         return false;
       }
       return true;
     }},

    // for op iter arg placed in same partition as op that produces
    // its value in the loop body (if it is not a token)
    {"for_op_iter_arg",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       if (from->getParent() != to->getParent())
         // skip if not both in the loop body
         return false;
       if (!isForIterArg(to))
         // skip is not to an iter arg
         return false;
       if (isa<AsyncTokenType>(to->getValue().getType()))
         // skip if a token type
         return false;
       return true;
     }},

    // for op iter arg placed in same partition as op that consumes
    // its value (if it is a token)
    {"for_op_iter_arg_token",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       if (!isForIterArg(from))
         // skip if not from an iter arg
         return false;
       if (!isa<AsyncTokenType>(from->getValue().getType()))
         // skip if not a token
         return false;
       return true;
     }},

    // if op result placed in same partition as MMA op that produces it (if it
    // is a token)
    {"if_op_result_token",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       if (!isMMA(from)) {
         // skip if not from an MMA
       }
       if (!isIfResult(to))
         // skip if not to an if op result
         return false;
       if (!isa<AsyncTokenType>(to->getValue().getType()))
         // skip if not a token
         return false;
       return true;
     }},

    // merge expensive SFU ops with their dependencies (except MMA, STORE and
    // other SFU)
    {"sfu_consumer",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return isCostlySFU(to) && !isMMA(from) && !isLoad(from) && !isSFU(from);
     }},

    // straight sequence of NONE ops merges together
    {"sequence",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       if (from->getNumOutDataEdges() > 1 || to->getNumInDataEdges() > 1)
         return false;
       return isNone(from) && isNone(to);
     }},

    // straight sequence of NONE op to SFU op merges together
    {"sequence_sfu",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       if (from->getNumOutDataEdges() > 1 || to->getNumInDataEdges() > 1)
         return false;
       return isNone(from) && isSFU(to);
     }},

    // NONE/cheap SFU merges with consumer (except LOAD, MMA or costly SFU)
    {"none_consumer",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return (isNone(from) || (isSFU(from) && !isCostlySFU(from))) &&
              !isNone(to) && !isMMA(to) && !isLoad(to) && !isCostlySFU(to);
     }},

    // NONE op with a single consumer merges together
    // {"single_consumer",
    //  [](Edge edge) {
    //    auto from = edge.getFromNode();
    //    auto to = edge.getToNode();
    //    if (from->getNumOutDataEdges() > 1)
    //      return false;
    //    return isNone(from) && isNone(to);
    //  }},

    // TMEM load merges with consumer
    // FIXME: limit to single consumer?
    {"tmem_load",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return node_isa<ttng::TMEMLoadOp>(from);
     }},

    // NONE merges with producer (except LOAD or MMA)
    {"none_producer",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return isNone(to) && !isNone(from) && !isMMA(from) && !isLoad(from);
     }},

    // TMEM partitions merge, if edge is high cost
    // {"tmem_high_cost",
    //  [](Edge edge) {
    //    auto from = edge.getFromNode();
    //    auto to = edge.getToNode();
    //    return isTMEM(from) && isTMEM(to) && edge.getSize() >= 256;
    //  }},

    // SFU merges with consumer
    //{"sfu_consumer",
    // [](Edge edge) {
    //   auto from = edge.getFromNode();
    //   auto to = edge.getToNode();
    //   return isSFU(from) && !isNone(to);
    // }},

    // // TMEM store op merges with uses
    // {"tmem_store",
    //  [](Edge edge) {
    //    auto from = edge.getFromNode();
    //    auto to = edge.getToNode();
    //    return node_isa<ttng::TMEMStoreOp>(to) && isNone(from);
    //  }},

    // // NONE/TMEM ops preceeding STORE merged together
    // {"store",
    //  [](Edge edge) {
    //    auto from = edge.getFromNode();
    //    auto to = edge.getToNode();
    //    return from->isOp() && to->isOp() && (isNone(from) || isTMEM(from)) &&
    //           isStore(to);
    //  }},

    // merge connected STORE partitions together
    // these are both using tt.descriptor_store and have a dataflow edge
    // between, so avoid communicating between partitions via aref
    {"connected_store",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return isStore(from) && isStore(to);
     }},

    // merge connected NONE partitions together
    {"connected_none",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return isOnlyNone(from) && isOnlyNone(to);
     }},

    // merge connected partitions together if edge between is expensive
    // TODO: this might be better expressed as a horizontal rule,
    // that aims to keep shmem usage under the limit
    {"connected",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return !isLoad(from) && !isLoad(to) && !isMMA(from) && !isMMA(to) &&
              edge.getSize() > 8192; // FIXME: seemingly arbitrary size...
     }},

    // // merge connected MMA partitions together
    // {"connected_mma",
    //  [](Edge edge) {
    //    auto from = edge.getFromNode();
    //    auto to = edge.getToNode();
    //    return isMMA(from) && isMMA(to);
    //  }},

    // // NONE ops following TMEM/STORE merged together
    // {"none_following",
    //  [](Edge edge) {
    //    auto from = edge.getFromNode();
    //    auto to = edge.getToNode();
    //    return (isTMEM(from) || isStore(from)) && isNone(to);
    //  }},
    //
    // // NONE group followed by STORE should merge
    // {"none_store",
    //  [](Edge edge) {
    //    auto from = edge.getFromNode();
    //    auto to = edge.getToNode();
    //    return isNone(from) && isStore(to);
    //  }},
    //
    // // NONE group followed by NONE should merge
    // {"none_none",
    //  [](Edge edge) {
    //    auto from = edge.getFromNode();
    //    auto to = edge.getToNode();
    //    return isNone(from) && isNone(to);
    //  }},
    //
    // // NONE group followed by LOAD should merge
    // {"none_load",
    //  [](Edge edge) {
    //    auto from = edge.getFromNode();
    //    auto to = edge.getToNode();
    //    return isNone(from) && isLoad(to);
    //  }},
    //
    // // NONE group followed by TMEM should merge
    // {"none_tmem",
    //  [](Edge edge) {
    //    auto from = edge.getFromNode();
    //    auto to = edge.getToNode();
    //    return isNone(from) && isTMEM(to);
    //  }},

};

SmallVector<std::pair<std::string, std::function<bool(Edge)>>> constraints = {
    // don't merge manual partitions
    {"manual",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return !(isManual(from) && isManual(to));
     }},

    // don't merge partitions with tmem ops into mma partitions
    {"tmem_mma",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return !((isMMA(from) && isTMEM(to)) || (isMMA(to) && isTMEM(from)));
     }},

    // don't merge tmem alloc (non-token form) into mma partition
    {"tmem_alloc",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return !(node_isa<ttng::TMEMAllocOp>(from) && isMMA(to));
     }},

    // don't merge partition with high cost, with other partitions with low
    // costs
    // {"cost",
    //  [](Edge edge) {
    //    auto from = edge.getFromNode();
    //    auto to = edge.getToNode();
    //    return !(node_isa<ttng::TMEMAllocOp>(from) && isMMA(to));
    //  }},
};

bool isCritical(Partition *partition) {
  SmallVector<Node *> stack;
  DenseSet<Node *> seen;
  for (auto node : partition->getNodes()) {
    stack.push_back(node);
    seen.insert(node);
  }

  while (!stack.empty()) {
    auto node = stack.back();
    stack.pop_back();
    if (!node->isData())
      continue;
    if (node->isOp() && isa<ttng::MMAv5OpInterface>(node->getOp()))
      return true;

    for (auto edge : node->getOutEdges()) {
      auto next_node = edge.getToNode();
      if (!seen.contains(next_node)) {
        stack.push_back(next_node);
        seen.insert(next_node);
      }
    }
  }

  return false;
};

SetVector<Partition *> getConsumingPartitions(Partition *partition) {
  SetVector<Partition *> result;
  for (auto node : partition->getNodes()) {
    for (auto edge : node->getOutEdges()) {
      auto outPartition = edge.getToNode()->getPartition();
      if (outPartition != partition)
        result.insert(outPartition);
    }
  }
  return result;
}

DenseSet<Partition *> getReachablePartitions(Partition *partition) {
  // look for all partitions connected to this partition via data edges
  DenseSet<Partition *> partitions;
  SmallVector<Node *> stack;
  DenseSet<Node *> seen;
  auto addNode = [&](Node *node) {
    if (!node)
      return;
    if (node->isData() && !seen.contains(node)) {
      partitions.insert(node->getPartition());
      stack.push_back(node);
      seen.insert(node);
    }
  };
  for (auto node : partition->getNodes())
    addNode(node);

  while (!stack.empty()) {
    auto node = stack.back();
    stack.pop_back();
    for (auto input : node->getInputs())
      addNode(input.getNode());
    for (auto outputs : node->getOutputs())
      for (auto output : outputs)
        addNode(output.getNode());
  }
  return partitions;
}

DenseSet<Operation *> getTMEMAllocs(Partition *partition) {
  // look for all tmem allocs used by the partition
  DenseSet<Operation *> result;
  for (auto node : partition->getNodes()) {
    if (!node->isOp())
      continue;
    Operation *alloc = nullptr;
    if (auto load = dyn_cast<ttng::TMEMLoadOp>(node->getOp())) {
      alloc = load.getOperand(0).getDefiningOp();
    }
    if (auto store = dyn_cast<ttng::TMEMStoreOp>(node->getOp())) {
      alloc = store.getOperand(0).getDefiningOp();
    }
    if (alloc) {
      assert(isa<ttng::TMEMAllocOp>(alloc));
      result.insert(alloc);
    }
  }
  return result;
}

SmallVector<
    std::pair<std::string, std::function<bool(Partition *, Partition *)>>>
    partition_heuristics = {
        // merge mma partitions
        {"mma",
         [](Partition *a, Partition *b) {
           auto a_is_mma = (a->getFlags() == Flags::MMA);
           auto b_is_mma = (b->getFlags() == Flags::MMA);
           return a_is_mma && b_is_mma;
         }},

        // merge load partitions
        {"load",
         [](Partition *a, Partition *b) {
           auto a_is_load = (a->getFlags() == Flags::LOAD);
           auto b_is_load = (b->getFlags() == Flags::LOAD);
           return a_is_load && b_is_load;
         }},

        // merge none with store partitions
        {"none",
         [](Partition *a, Partition *b) {
           auto a_is_none = (a->getFlags() == Flags::NONE);
           auto b_is_none = (b->getFlags() == Flags::NONE);
           auto a_is_store = (a->getFlags() & Flags::STORE);
           auto b_is_store = (b->getFlags() & Flags::STORE);
           return (a_is_none && b_is_store) || (a_is_store && b_is_none);
         }},

        // merge store partitions
        // {"store",
        //  [](Partition *a, Partition *b) {
        //    auto a_is_store = (a->getFlags() & Flags::STORE) ||
        //                      (a->getFlags() & Flags::TMEM) ||
        //                      (a->getFlags() == Flags::NONE);
        //    auto b_is_store = (b->getFlags() & Flags::STORE) ||
        //                      (b->getFlags() & Flags::TMEM) ||
        //                      (b->getFlags() == Flags::NONE);
        //    return a_is_store && b_is_store;
        //  }},

        // merge TMEM partitions together, if they use the same tmem alloc
        // and that alloc is used in more than 2 partitions
        // as aref does not support tmem with more than 2 partitions
        // FIXME: this is a bit broken - it might merge too much - i.e. doesn't
        // just merge up to 2 partitions
        {"tmem",
         [](Partition *a, Partition *b) {
           auto a_is_tmem = (a->getFlags() & Flags::TMEM);
           auto b_is_tmem = (b->getFlags() & Flags::TMEM);
           if (!a_is_tmem || !b_is_tmem) {
             return false;
           }
           auto allocs_a = getTMEMAllocs(a);
           auto allocs_b = getTMEMAllocs(b);
           // if the sets are overlapping
           bool overlap = false;
           for (auto alloc_a : allocs_a) {
             if (allocs_b.contains(alloc_a)) {
               overlap = true;
               break;
             }
           }
           if (!overlap)
             return false;
           return true;
         }},
};

void mergePartitions(Graph *graph, std::string funcName,
                     VisualizationInfo &vis_info) {
  // Note: this implementation is slow. It can be improved by incrementally
  // updating the data structures rather than rebuilding the whole lot when a
  // rule is applied
  const auto &options = get_options();
  LLVM_DEBUG({ llvm::errs() << "#### applying heuristics...\n"; });
  int iter = 0;
  bool changed = false;
  do {
    changed = false;

    auto crossingEdges = getCrossingEdges(graph);
    LLVM_DEBUG({
      llvm::errs() << "\n"
                   << crossingEdges.size() << " crossing edges remaining\n";
    });

    for (auto [name, apply] : heuristics) {
      for (auto edge : crossingEdges) {
        if (apply(edge)) {

          // check if applying the heuristic will observe the contraints
          bool ok = true;
          // Note: constraints just prevent epilogue partition ops merging
          // into mma partitions, so disable them if we don't want epilogue
          // partitions
          for (auto [name, constraint] : constraints) {
            if (!constraint(edge)) {
              ok = false;
              break;
            }
          }
          if (ok) {
            LLVM_DEBUG({
              llvm::errs() << "\napply heuristic \"" << name << "\"\n";
              llvm::errs() << edge.getFromNode()->getLabel() << " -> "
                           << edge.getToNode()->getLabel() << "\n";
              llvm::errs() << "partitions "
                           << edge.getFromNode()->getPartition() << " -> "
                           << edge.getToNode()->getPartition() << "\n";
              llvm::errs() << "flags "
                           << edge.getFromNode()->getPartition()->getFlags()
                           << " -> "
                           << edge.getToNode()->getPartition()->getFlags()
                           << "\n";
            });

            // merge the partitions
            auto from_partition = edge.getFromNode()->getPartition();
            auto to_partition = edge.getToNode()->getPartition();
            Partition::merge(from_partition, to_partition);

            if (options.dump_dot) {
              visualize(funcName, "merge-step",
                        std::string("merge: rule ") + name, graph, vis_info);
            }
            iter++;

            changed = true;
            break;
          }
        }
      }
      if (changed)
        break;
    }
  } while (changed);

  visualize(funcName, "merge-step", "edge based merge complete", graph,
            vis_info);

  {
    // look at every pair of partitions and check if they should be merged
    auto merge_partitions_step = [&]() {
      SmallVector<Partition *> all_partitions;
      for (auto &partition : graph->getPartitions()) {
        all_partitions.push_back(partition.get());
      }
      for (auto [name, apply] : partition_heuristics) {
        for (auto partitionA : all_partitions) {
          for (auto partitionB : all_partitions) {
            if (partitionA == partitionB || partitionA->empty() ||
                partitionB->empty())
              continue;
            if (apply(partitionA, partitionB)) {
              LLVM_DEBUG({
                llvm::errs() << "\nmerge \"" << name << "\" ----\n";
                partitionA->dump();
                partitionB->dump();
              });
              Partition::merge(partitionA, partitionB);
              if (options.dump_dot) {
                std::stringstream filename;
                visualize(funcName, "merge-step",
                          std::string("merge: rule ") + name, graph, vis_info);
              }
              iter++;
              return false;
            }
          }
        }
      }
      return true;
    };

    while (true) {
      if (merge_partitions_step())
        break;
    }
  }

  visualize(funcName, "merge-step", "partition based merge complete", graph,
            vis_info);

  // push broadcast ops into consumer partition, to reduce shared memory
  // pressure
  // FIXME: doesn't actually do anything, just checks this is the case
  // as the heuristics should guarantee this anyway
  // {
  //   bool changed = false;
  //   do {
  //     changed = false;
  //     auto crossingEdges = getCrossingEdges(graph);
  //     for (auto edge : crossingEdges) {
  //       auto from = edge.getFromNode();
  //       if (!from->isOp())
  //         continue;
  //       auto op = from->getOp();
  //       if (isa_and_nonnull<tt::BroadcastOp, tt::ExpandDimsOp>(op))
  //         assert(false &&
  //                "FIXME: push broadcast/expand dims into previous
  //                partition");
  //     }
  //   } while (changed);
  // }

  LLVM_DEBUG({ llvm::errs() << "\n#### heuristics done\n"; });
}

void propagatePartitions(Graph *graph, std::string funcName,
                         VisualizationInfo &vis_info) {
  auto &options = get_options();

  if (options.dump_dot)
    visualize(funcName, "propagate", "before propagate", graph, vis_info);

  // propagate partitions to parent ops
  SmallVector<Node *> leaves;

  graph->walk([&](Node *node) {
    // node is a leaf if it has a region,
    // and none of the ops in the region are leaves
    bool is_leaf = !node->getNodes().empty();
    for (auto &child : node->getNodes()) {
      if (!child->getNodes().empty()) {
        is_leaf = false;
        break;
      }
    }
    if (is_leaf)
      leaves.push_back(node);
  });

  bool changed = true;
  while (changed) {
    for (auto leaf : leaves) {
      // partitions for leaf are union of partitions of all ops contained in
      // the leaf
      SetVector<Partition *> partitions;
      for (auto &node : leaf->getNodes())
        partitions.insert(node->getPartitions().begin(),
                          node->getPartitions().end());
      leaf->addPartitions(partitions);

      // propagate to parent nodes
      auto node = leaf->getParent();
      while (node) {
        // include union of partitions of ops in the parent
        for (auto &child : node->getNodes())
          partitions.insert(child->getPartitions().begin(),
                            child->getPartitions().end());
        node->addPartitions(partitions);
        node = node->getParent();
      }
    }

    // propagate partitions to non-data nodes
    {
      SmallVector<Node *> nodes;
      // include nodes with regions
      graph->walk([&](Node *node) {
        if (!node->getNodes().empty())
          nodes.push_back(node);
      });
      // include data nodes
      for (auto &partition : graph->getPartitions())
        for (auto &node : partition->getNodes())
          if (node->isData())
            nodes.push_back(node);

      changed = false;
      for (auto node : nodes) {
        SmallVector<Node *> stack;
        DenseSet<Node *> seen;
        auto partitions = node->getPartitions();
        stack.push_back(node);
        seen.insert(node);

        while (!stack.empty()) {
          auto node = stack.back();
          stack.pop_back();

          auto propagate = [&](Edge edge, Node *node) {
            if (!node || node->isData())
              return;
            auto numPartitionsBefore = node->getPartitions().size();
            node->addPartitions(partitions);
            auto numPartitionsAfter = node->getPartitions().size();
            changed |= (numPartitionsBefore != numPartitionsAfter);
            if (seen.count(node) == 0) {
              stack.push_back(node);
              seen.insert(node);
            }
          };

          for (auto edge : node->getInEdges())
            propagate(edge, edge.getFromNode());
          // for (auto edge : node->getOutEdges())
          //  propagate(edge, edge.getToNode());
        }
      }
    }
  }

  if (options.dump_dot)
    visualize(funcName, "propagate", "after propagate", graph, vis_info);

  // propagate partitions to non-data nodes (forward)
  {
    SmallVector<Node *> nodes;
    // get nodes that have no partition assigned
    graph->walk([&](Node *node) {
      if (!node->hasPartition())
        nodes.push_back(node);
    });

    changed = false;
    while (!nodes.empty()) {
      // try propagating partitions forward to nodes with no partition
      int start_size = nodes.size();
      bool changed = false;
      for (auto node : nodes) {
        for (auto edge : node->getInEdges()) {
          if (!edge.getFromNode())
            continue;
          if (edge.getFromNode()->hasPartition()) {
            for (auto partition : edge.getFromNode()->getPartitions())
              node->setPartition(partition);
            changed = true;
          }
        }
      }
      // must have changed something, otherwise infinite loop
      // assert(changed);
      // remove all nodes that now have a partition
      nodes.erase(
          std::remove_if(nodes.begin(), nodes.end(),
                         [](Node *node) { return node->hasPartition(); }),
          nodes.end());
      int end_size = nodes.size();
      if (start_size == end_size) {
        // no change -> exit
        break;
      }
    }
  }

  if (options.dump_dot)
    visualize(funcName, "propagate", "propagate forward", graph, vis_info);

  // propagate partitions of tt.reduce into its body
  graph->walk([&](Node *node) {
    if (node->isOp() && isa<tt::ReduceOp>(node->getOp())) {
      auto partitions = node->getPartitions();
      node->walk(
          [&](Node *child_node) { child_node->addPartitions(partitions); });
    }
  });

  if (options.dump_dot)
    visualize(funcName, "propagate", "propagate reduce", graph, vis_info);

  // Corner case: tmem store following tmem alloc should be in a warp
  // partition with 4 warps (i.e. a non-mma partition)
  // This fixes the case where in a tmem alloc + initial store that feeds into
  // an mma, the store is propagated the partition of the mma. It should instead
  // have the same partition as the alloc
  SmallVector<Node *> patched_nodes;

  graph->walk([&](Node *node) {
    if (node->isData() || !node->isOp() ||
        !isa<ttng::TMEMStoreOp>(node->getOp())) {
      return;
    }

    Node *alloc = nullptr;
    for (auto edge : node->getInEdges()) {
      if (edge.getToIdx() == 1) { // token edge
        alloc = edge.getFromNode();
        break;
      }
    }
    if (!alloc || !alloc->isOp() || !isa<ttng::TMEMAllocOp>(alloc->getOp()))
      return;

    // pick the first non-mma partition
    // does nothing if the only partitions are mma
    auto partitions = alloc->getPartitions();
    for (auto partition : partitions) {
      if (partition->getFlags() & MMA)
        continue;
      node->setPartition(partition);
      patched_nodes.push_back(node);
      break;
    }
  });

  if (options.dump_dot)
    visualize(funcName, "propagate", "tmem store corner case", graph, vis_info);

  // propagate partitions for patched up nodes to non-data nodes
  for (auto node : patched_nodes) {
    SmallVector<Node *> stack;
    DenseSet<Node *> seen;
    auto partitions = node->getPartitions();
    stack.push_back(node);
    seen.insert(node);

    while (!stack.empty()) {
      auto node = stack.back();
      stack.pop_back();

      for (auto edge : node->getInEdges()) {
        if (edge.isDataValue())
          continue;
        auto fromNode = edge.getFromNode();
        if (!fromNode)
          continue;
        fromNode->addPartitions(partitions);

        if (seen.count(edge.getFromNode()) == 0) {
          stack.push_back(fromNode);
          seen.insert(fromNode);
        }
      }
    }
  }
}

void duplicateCheapOps(Graph *graph, std::string funcName,
                       VisualizationInfo &vis_info) {
  auto &options = get_options();

  if (options.dump_dot)
    visualize(funcName, "duplicate", "before duplicate cheap ops", graph,
              vis_info);

  // for each partition:
  // look at all crossing edges leaving the partition
  // do a depth first search through NONE nodes, if we hit the same partition
  // assign all nodes on that path to the partition
  for (auto &partition : graph->getPartitions()) {

    auto crossingEdges = getOutCrossingEdges(partition.get());

    for (auto edge : crossingEdges) {
      // only handle start nodes with a single partition
      if (edge.getFromNode()->getPartitions().size() != 1)
        continue;
      auto startPartition = edge.getFromNode()->getPartition();

      // only handle nodes with a single partition
      auto start = edge.getToNode();
      if (start->getPartitions().size() != 1)
        continue;
      auto partition = start->getPartition();

      auto isCandidate = [](Node *node) {
        // FIXME: ignore costly SFU
        return (getNodeFlags(node) == Flags::NONE ||
                getNodeFlags(node) == Flags::SFU);
      };

      if (!isCandidate(edge.getToNode())) {
        continue;
      }

      // llvm::errs() << "\n\ntry\n";
      // edge.getFromNode()->dump();
      // edge.getToNode()->dump();

      auto update = [&]() {
        std::map<Node *, Node *> parentMap;

        SmallVector<Node *> stack;
        stack.push_back(start);
        DenseSet<Node *> seen;

        while (!stack.empty()) {
          auto node = stack.back();
          // llvm::dbgs() << "visit\n";
          // node->dump();
          stack.pop_back();
          if (!seen.contains(node)) {
            seen.insert(node);
            for (auto edge : node->getOutEdges()) {
              auto child = edge.getToNode();
              if (!seen.contains(child)) {
                // llvm::dbgs() << "child\n";
                // child->dump();
                if (child->getPartitions().size() != 1 || !isCandidate(child)) {
                  // llvm::dbgs() << "no match, ignore path\n";
                } else if (child->getPartition() == partition) {
                  // llvm::dbgs() << "same partition, follow...\n";
                  parentMap.emplace(child, node);
                  stack.push_back(child);
                } else if (child->getPartition() == startPartition) {
                  // llvm::dbgs() << "HIT!\n";
                  // found a path, set all nodes on the path to the partition
                  // llvm::dbgs() << "set partition\n";
                  node->addPartition(startPartition);
                  // node->dump();
                  while (parentMap.find(node) != parentMap.end()) {
                    node = parentMap[node];
                    node->addPartition(startPartition);
                    // node->dump();
                  }

                  if (options.dump_dot)
                    visualize(funcName, "duplicate", "duplicate cheap ops",
                              graph, vis_info);

                  return;
                } else {
                  // llvm::dbgs() << "no match, ignore path\n";
                }
              }
            }
          }
        }
      };
      update();
    }
  }

  if (options.dump_dot)
    visualize(funcName, "duplicate", "duplicate cheap ops done", graph,
              vis_info);
}

void visualize(std::string key, std::string filename, std::string title,
               Graph *graph, VisualizationInfo &info) {

  static std::map<std::string, int> keys;
  if (keys.find(key) == keys.end()) {
    keys[key] = 0;
  }
  auto idx = keys[key];
  keys[key]++;

  const auto &options = get_options();

  std::stringstream path;
  path << "graph-" << key << "-" << std::setfill('0') << std::setw(4) << idx
       << "-" << filename << ".dot";
  std::ofstream dot(path.str());
  dot << "digraph G {\n";
  dot << "label = \"" << title << "\";\n";
  dot << "labelloc=\"t\";\n";
  dot << "labeljust=\"c\";\n";

  DenseMap<Node *, size_t> node_ids;

  auto getPartitionId = [&](Partition *partition) {
    if (info.partition_ids.count(partition) == 0)
      info.partition_ids[partition] = info.partition_ids.size();
    return info.partition_ids[partition];
  };

  auto getPartitionColor = [&](Partition *partition) {
    if (info.partition_colors.count(partition) == 0) {
      size_t color = info.partition_colors.size() + 1;
      color = (color % 12) + 1;
      info.partition_colors[partition] =
          std::string("/set312/") + std::to_string(color);
    }
    return info.partition_colors[partition];
  };

  // reset colors
  // info.partition_colors.clear();
  // for (auto &partition : graph->getPartitions()) {
  //   if (partition->empty())
  //     continue;
  //   getPartitionColor(partition.get());
  // }

  // add nodes
  std::function<void(Node *)> visitNodes = [&](Node *graph) {
    for (auto &node_obj : graph->getNodes()) {
      auto node = node_obj.get();

      if (options.dump_data_only && !node->isData() && !node->containsData())
        // skip if dumping data nodes only, and this op is non-data or doesn't
        // contain a data node
        continue;
      if (options.dump_loop_only && !node->inLoopBody() &&
          !node->containsLoopBody())
        // skip if dumping loop body nodes only
        continue;

      node_ids[node] = node_ids.size();

      if (!node->getNodes().empty())
        dot << "subgraph cluster_cx" << node_ids[node] << " {\n"
            << "label=\"\"\n";
      dot << "x" << node_ids[node] << "[shape=plaintext, ";
      if (node->isData())
        dot << "color=blue, ";
      dot << "label=<";
      dot << "<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\">";
      if (node->getNumInputs() > 1) {
        dot << "<TR>";
        for (size_t idx = 0; idx < node->getNumInputs(); idx++)
          dot << "<TD PORT=\"in" << idx << "\">" << idx << "</TD>";
        dot << "</TR>";
      }
      dot << "<TR><TD PORT=\"inout\"";
      size_t colspan = std::max(node->getNumInputs(), node->getNumOutputs());
      if (colspan > 0)
        dot << " COLSPAN=\"" << colspan << "\"";
      dot << ">";

      dot << "<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\"><TR>";
      if (node->hasPartition()) {
        for (auto partition : node->getPartitions()) {
          auto name = std::to_string(getPartitionId(partition));
          dot << "<TD BGCOLOR=\"" << getPartitionColor(partition) << "\">"
              << name << "{" << partition->getCost() << "}"
              << "[" << partition->getFlags() << "]</TD>";
        }
      }
      dot << "<TD>" << node->getLabel();
      if (node->isData())
        dot << " [" << getNodeFlags(node) << "]";
      dot << "</TD></TR></TABLE>";
      dot << "</TD></TR>";

      if (node->hasCost()) {
        dot << "<TR><TD";
        if (colspan > 0)
          dot << " COLSPAN=\"" << colspan << "\"";
        dot << ">";
        dot << "cost:" << node->getCost();
        dot << "</TD></TR>";
      }

      if (node->getNumOutputs() > 1) {
        dot << "<TR>";
        for (size_t idx = 0; idx < node->getNumOutputs(); idx++)
          dot << "<TD PORT=\"out" << idx << "\">" << idx << "</TD>";
        dot << "</TR>";
      }
      dot << "</TABLE>>];\n";
      if (!node->getNodes().empty()) {
        visitNodes(node);
        dot << "}\n";
      }
    }
  };
  visitNodes(graph->getRoot());

  // add edges
  std::function<void(Node *)> visitEdges = [&](Node *node) {
    size_t idx = 0;
    for (auto inputPorts : node->getOutputs()) {
      OutputPort outputPort{node, idx};
      for (auto inputPort : inputPorts) {
        Edge edge(outputPort, inputPort);
        if (node_ids.count(outputPort.getNode()) == 0 ||
            node_ids.count(inputPort.getNode()) == 0)
          continue;
        dot << "x" << node_ids[outputPort.getNode()];
        dot << ":";
        if (outputPort.getNode()->getNumOutputs() == 1)
          dot << "inout";
        else
          dot << "out" << outputPort.getIdx();
        dot << " -> ";
        dot << "x" << node_ids[inputPort.getNode()];
        dot << ":";
        if (inputPort.getNode()->getNumInputs() == 1)
          dot << "inout";
        else
          dot << "in" << inputPort.getIdx();
        std::vector<std::string> attrs;
        if (edge.isDataValue()) {
          if (edge.getFromNode()->getPartitions().size() > 1 ||
              edge.getToNode()->getPartitions().size() > 1)
            // invalid edge, should only have one partition
            attrs.push_back("color=\"green\"");
          else if (edge.crossesPartitions())
            attrs.push_back("color=\"red\"");
          else
            attrs.push_back("color=\"blue\"");
          auto size = edge.getSize();
          if (size != 1) {
            attrs.push_back("label=\"" + std::to_string(size) + "\"");
          }
        }
        if (!attrs.empty()) {
          dot << "[";
          for (auto attr = attrs.begin(); attr != attrs.end(); attr++) {
            if (attr != attrs.begin()) {
              dot << ",";
            }
            dot << *attr;
          }
          dot << "]";
        }
        dot << ";\n";
      }
      idx++;
    }
    for (auto &node : node->getNodes())
      visitEdges(node.get());
  };
  visitEdges(graph->getRoot());

  dot << "}\n";
}

bool isSimpleLoad(Partition *partition) {
  for (auto node : partition->getNodes()) {
    if (!node->isData() || !node->isOp())
      continue;
    auto op = node->getOp();
    // Note: we do not consider cp.async (tt.Load) a simple load,
    // as it needs >1 warp
    if (!isa<tt::DescriptorLoadOp, ttg::LocalAllocOp>(op))
      return false;
  }
  return true;
}

bool isSimpleMMA(Partition *partition) {
  for (auto node : partition->getNodes()) {
    if (!node->isData() || !node->isOp())
      continue;
    auto op = node->getOp();
    if (!isa<ttng::MMAv5OpInterface, ttg::MemDescTransOp>(op))
      return false;
  }
  return true;
}

void serialize(size_t idx, Operation *region, Graph *graph) {
  auto context = graph->getRoot()->getOp()->getContext();
  Builder b(context);

  // annotate loop with index
  region->setAttr(kWarpSpecializeTagAttrName, b.getI32IntegerAttr(idx));

  auto setPartitionsAttr = [&](Operation *op, Node *node) {
    // not for func op
    if (isa<tt::FuncOp>(op))
      return;
    SmallVector<int> partitions;
    for (auto partition : node->getPartitions())
      partitions.push_back(*partition->id);
    std::sort(partitions.begin(), partitions.end());
    auto partitionsAttr = b.getDenseI32ArrayAttr(partitions);
    op->setAttr(kPartitionAttrName, partitionsAttr);
    // set same paritions in yield ops
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      cast<scf::YieldOp>(forOp.getBody()->getTerminator())
          ->setAttr(kPartitionAttrName, partitionsAttr);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      ifOp.thenYield()->setAttr(kPartitionAttrName, partitionsAttr);
      if (!ifOp.getElseRegion().empty()) {
        ifOp.elseYield()->setAttr(kPartitionAttrName, partitionsAttr);
      }
    }
    // FIXME: handle ReduceOp
  };

  auto setPartitionOutputsAttr = [&](Operation *op, size_t idx, size_t size,
                                     Node *node) {
    llvm::SmallVector<Attribute> partitionAttrs;
    if (op->hasAttr(kPartitionOutputsAttrName)) {
      // get existing partitions
      for (auto attr :
           op->getAttrOfType<ArrayAttr>(kPartitionOutputsAttrName)) {
        partitionAttrs.push_back(attr);
      }
      assert(partitionAttrs.size() == size);
    } else {
      // initialize to no partitions
      for (size_t i = 0; i < size; i++)
        partitionAttrs.push_back(b.getDenseI32ArrayAttr({}));
    }

    // update partitions for this output
    SmallVector<int> partitions;
    for (auto partition : node->getPartitions())
      partitions.push_back(*partition->id);
    std::sort(partitions.begin(), partitions.end());
    partitionAttrs[idx] = b.getDenseI32ArrayAttr(partitions);
    op->setAttr(kPartitionOutputsAttrName,
                ArrayAttr::get(context, partitionAttrs));
  };

  graph->walk([&](Node *node) {
    if (node->isOp()) {
      setPartitionsAttr(node->getOp(), node);

      if (auto ret = dyn_cast<tt::ReduceReturnOp>(node->getOp())) {
        // result of a reduce
        auto reduce = node->getParent()->getOp();
        setPartitionOutputsAttr(reduce, 0, 1, node);
      }

    } else {
      auto value = node->getValue();
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        auto parentOp = blockArg.getOwner()->getParentOp();
        if (isa<tt::FuncOp>(parentOp)) {
          // nothing for func ops
        } else if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
          if (blockArg.getArgNumber() == 0) {
            // nothing for induction variable
          } else {
            // for op iter args
            setPartitionOutputsAttr(parentOp, blockArg.getArgNumber() - 1,
                                    forOp.getResultTypes().size(), node);
          }
        } else {
          assert(false);
        }
      } else if (auto result = dyn_cast<OpResult>(value)) {
        auto op = result.getOwner();
        if (isa<scf::ForOp>(op)) {
          // do nothing (handled by block arg)
        } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          // result of an if
          setPartitionOutputsAttr(op, result.getResultNumber(),
                                  ifOp.getResultTypes().size(), node);
        } else {
          assert(false);
        }
      } else {
        assert(false);
      }
    }
  });

  // set stages
  SmallVector<Attribute> stages;
  for (auto &partition : graph->getPartitions()) {
    if (partition->empty())
      continue;
    auto id = *partition->id;
    while (id >= stages.size())
      stages.push_back(b.getI32IntegerAttr(0));
    stages[id] = b.getI32IntegerAttr(partition->getStage());
  }
  region->setAttr(kPartitionStagesAttrName, b.getArrayAttr(stages));
}

bool hasFlattenedEpilogue(Operation *op) {
  SmallVector<ttng::TMEMLoadOp> tmemLoadOps;
  op->walk([&](ttng::TMEMLoadOp op) { tmemLoadOps.push_back(op); });
  for (auto tmemLoadOp : tmemLoadOps)
    if (auto tok = tmemLoadOp.getDep())
      if (auto op = tok.getDefiningOp())
        if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(op)) {
          auto mmaLoop = mmaOp->getParentOfType<scf::ForOp>();
          auto loadIfOp = tmemLoadOp->getParentOfType<scf::IfOp>();
          if (mmaLoop && loadIfOp && mmaLoop.getBody() == loadIfOp->getBlock())
            return true;
        }
  return false;
}

SmallVector<SmallVector<mlir::Operation *>>
duplicateViewOps(Operation *region) {
  // Ensure all view ops/broadcast/expand dims have a single user, by
  // duplicating them where necessary. Ensures these ops do not span more
  // than one partition
  // Note: Duplicated ops within a single partition are deduplicated after
  // analysis

  SmallVector<SmallVector<mlir::Operation *>> duplicatedViewOps;
  SmallVector<mlir::Operation *> viewOps;

  region->walk([&](mlir::Operation *op) {
    if (isViewOp(op))
      viewOps.push_back(op);
  });

  while (!viewOps.empty()) {
    auto op = viewOps.pop_back_val();

    assert(op->getResults().size() == 1);
    auto result = op->getResult(0);

    bool first = true;
    for (auto &use : result.getUses()) {
      if (first) {
        duplicatedViewOps.push_back({op});
      } else {
        auto newOp = OpBuilder(op).clone(*op);
        use.set(newOp->getResult(0));
        duplicatedViewOps.back().push_back(newOp);
      }
      first = false;
    }
  }
  return duplicatedViewOps;
}

void deduplicateViewOps(
    Operation *region,
    SmallVector<SmallVector<mlir::Operation *>> duplicatedViewOps) {

  // get partition assignments for the duplicated ops, and
  // re-merge those that have the same partition assignment
  for (auto ops : duplicatedViewOps) {
    std::map<SmallVector<int>, SmallVector<mlir::Operation *>> partitionedOps;
    for (auto op : ops) {
      assert(op->hasAttr(kPartitionAttrName));
      auto partitionsRef =
          cast<DenseI32ArrayAttr>(op->getAttr(kPartitionAttrName)).asArrayRef();
      SmallVector<int> partitions(partitionsRef.begin(), partitionsRef.end());
      if (partitionedOps.find(partitions) == partitionedOps.end()) {
        partitionedOps[partitions] = {};
      }
      partitionedOps[partitions].push_back(op);
    }
    for (auto partition : partitionedOps) {
      auto &sameOps = partition.second;
      if (sameOps.size() <= 1)
        continue;
      auto op = sameOps.front();
      for (auto it = sameOps.begin() + 1; it != sameOps.end(); it++) {
        // merge the two ops
        (*it)->replaceAllUsesWith(op->getResults());
        (*it)->erase();
      }
    }
  }
}

void assignPartitionIds(Graph *graph) {
  // assign unique ids for partitions
  // starting with store partitions, followed by everything else
  // FIXME: this is a hack, why is ordering important?
  size_t idx = 0;

  SmallVector<Partition *> store_partitions;
  SmallVector<Partition *> mma_partitions;
  SmallVector<Partition *> load_partitions;
  SmallVector<Partition *> other_partitions;

  for (auto &partition : graph->getPartitions()) {
    if (partition->empty())
      continue;
    if (partition->getFlags() & Flags::STORE)
      store_partitions.push_back(partition.get());
    else if (partition->getFlags() & Flags::MMA)
      mma_partitions.push_back(partition.get());
    else if (partition->getFlags() & Flags::LOAD)
      load_partitions.push_back(partition.get());
    else
      other_partitions.push_back(partition.get());
  }

  for (auto partition : other_partitions) {
    partition->id = idx;
    idx++;
  }
  for (auto partition : store_partitions) {
    partition->id = idx;
    idx++;
  }
  // ensure MMA and LOAD partitions are never the same as the default
  // partition
  if (idx == 0)
    idx++;
  for (auto partition : mma_partitions) {
    partition->id = idx;
    idx++;
  }
  for (auto partition : load_partitions) {
    partition->id = idx;
    idx++;
  }
}

void assignDefaultPartitions(Graph *graph) {
  // nodes with no partition placed in same partition as other ops in the
  // region or default partition if none. Note: we can't just use partitions
  // of parent op, as this includes things like tmem tokens
  Partition *defaultPartition = nullptr;
  for (auto &partition : graph->getPartitions()) {
    if (partition->id && *partition->id == 0) {
      defaultPartition = partition.get();
    }
  }
  graph->walk([&](Node *node) {
    if (node->getPartitions().empty()) {
      bool done = false;
      auto parent = node->getParent();
      if (parent && parent->isOp()) {
        for (auto &otherNode : parent->getNodes()) {
          if (node == otherNode.get()) {
            continue;
          }
          if (otherNode->isOp() && otherNode->hasPartition()) {
            node->addPartitions(otherNode->getPartitions());
            done = true;
          }
        }
      }
      if (!done) {
        assert(defaultPartition != nullptr);
        node->setPartition(defaultPartition);
      }
    }
  });
}

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct PartitionScheduling
    : public impl::TritonGPUPartitionSchedulingBase<PartitionScheduling> {
  using TritonGPUPartitionSchedulingBase::TritonGPUPartitionSchedulingBase;

  void runOnOperation() override {
    // find ops to partition; either:
    //   - all scf::For ops marked with "tt.warp_specialize"
    //   - or entire module if marked with "tt.warp_specialize"
    SmallVector<Operation *> ops;
    getOperation().walk([&](scf::ForOp op) {
      if (op->hasAttr(kWarpSpecializeAttrName))
        ops.push_back(op);
    });
    if (ops.empty() && getOperation()->hasAttr(kWarpSpecializeAttrName)) {
      ops.push_back(getOperation());
    }

    // run partitioner on each op
    size_t idx = 0;
    for (auto op : ops) {
      analyze(idx, op);
      idx++;
    }
  }

private:
  void analyze(size_t idx, Operation *op) {
    auto duplicatedOps = duplicateViewOps(op);
    tt::FuncOp func;
    if (auto m = dyn_cast<ModuleOp>(op)) {
      func = cast<tt::FuncOp>(m.getRegion().front().front());
    } else {
      func = op->getParentOfType<tt::FuncOp>();
      assert(func);
    }

    auto toInt = [](int dflt, std::string value) {
      if (value.size() == 0)
        return dflt;
      return std::stoi(value);
    };

    auto &options = get_options();
    options.dump_dot =
        tools::getBoolEnv("TRITON_PARTITION_SCHEDULING_ENABLE_DUMP_DOT");
    options.dump_data_only =
        tools::getBoolEnv("TRITON_PARTITION_SCHEDULING_DUMP_DATA_ONLY");
    options.dump_loop_only =
        tools::getBoolEnv("TRITON_PARTITION_SCHEDULING_DUMP_LOOP_ONLY");

    auto graph = buildGraph(op);
    auto initValues = initialDataValues(graph.get());
    propagateDataValues(initValues);
    options.manual = deserializeManualPartitions(op, graph.get());
    VisualizationInfo vis_info;
    auto key = func.getSymName().str() + "_" + std::to_string(idx);
    if (options.dump_dot)
      visualize(key, "input", "input", graph.get(), vis_info);
    initialPartitionAssignment(graph.get());
    if (options.dump_dot)
      visualize(key, "initial", "initial partitions", graph.get(), vis_info);
    mergePartitions(graph.get(), key, vis_info);
    if (options.dump_dot)
      visualize(key, "merge", "merged", graph.get(), vis_info);
    propagatePartitions(graph.get(), key, vis_info);
    if (options.dump_dot)
      visualize(key, "propagate", "propagated", graph.get(), vis_info);

    assignPartitionIds(graph.get());
    if (options.dump_dot)
      visualize(key, "assign-partition-ids", "assign partition ids",
                graph.get(), vis_info);
    // Handle case where ops with no uses (like llvm.intr.assume) get no
    // partition Assign them to default partition, and rerun propagation
    assignDefaultPartitions(graph.get());
    if (options.dump_dot)
      visualize(key, "assign-default", "assign default", graph.get(), vis_info);
    propagatePartitions(graph.get(), key, vis_info);
    if (options.dump_dot)
      visualize(key, "propagate", "propagated", graph.get(), vis_info);

    // FIXME: optimization - looks for paths of NONE ops, from one partition,
    // through another partition, and back to the same partition. Duplicate
    // these to avoid the copying involved (i.e. assign to both partitions)
    duplicateCheapOps(graph.get(), key, vis_info);

    if (options.dump_dot)
      visualize(key, "final", "final", graph.get(), vis_info);

    LLVM_DEBUG({
      llvm::errs() << "\nfinal partitions:\n";
      for (auto &partition : graph->getPartitions()) {
        if (partition->empty())
          continue;
        partition->dump();
      }
    });

    serialize(idx, op, graph.get());
    deduplicateViewOps(op, duplicatedOps);
    // FIXME: dedup removes ops, so following visualization can crash
    // if (options.dump_dot)
    //   visualize(key, "dedup-views", "dedup views", graph.get(), vis_info);
  }
};

} // namespace mlir::triton::gpu
