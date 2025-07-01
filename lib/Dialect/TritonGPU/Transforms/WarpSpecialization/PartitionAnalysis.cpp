#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/Debug.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace {

using namespace mlir;
using namespace triton;

namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

class Node;

enum Flags : uint8_t {
  NONE = 0,
  LOAD = 1 << 0,
  STORE = 1 << 1,
  MMA = 1 << 2,
  SIMT = 1 << 3,
};

Flags &operator|=(Flags &lhs, Flags rhs) {
  return lhs = static_cast<Flags>(lhs | rhs);
}

std::ostream &operator<<(std::ostream &stream, Flags flags) {
  std::vector<std::string> strs;
  if (flags == Flags::NONE) {
    strs.push_back("NONE");
  } else {
    if (flags & Flags::LOAD) {
      strs.push_back("LOAD");
    }
    if (flags & Flags::STORE) {
      strs.push_back("STORE");
    }
    if (flags & Flags::MMA) {
      strs.push_back("MMA");
    }
    if (flags & Flags::SIMT) {
      strs.push_back("SIMT");
    }
  }
  for (size_t i = 0; i < strs.size(); i++) {
    if (i != 0) {
      stream << "|";
    }
    stream << strs[i];
  }
  return stream;
}

class Group {
public:
  void add(Node *node);
  void remove(Node *node) { nodes.remove(node); }
  Flags getFlags() const { return flags; }
  const SetVector<Node *> &getNodes() const { return nodes; }

  static void merge(Group *lhs, Group *rhs);

private:
  Flags flags = Flags::NONE;
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

namespace llvm {
template <> struct DenseMapInfo<Port> {
  static inline Port getEmptyKey() { return {}; }

  static inline Port getTombstoneKey() { return Port(0, 1); }

  static unsigned getHashValue(const Port &port) {
    return std::hash<Node *>()(port.getNode()) ^
           std::hash<size_t>()(port.getIdx());
  }

  static bool isEqual(const Port &lhs, const Port &rhs) { return lhs == rhs; }
};
} // namespace llvm

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
  bool crossesGroups() const;
  Type getType() const;

private:
  OutputPort from;
  InputPort to;
};

class Node {
public:
  explicit Node(Operation *op) : op(op) {}

  Node(Node *parent, Operation *op, size_t numInputs, size_t numOutputs)
      : parent(parent), op(op) {
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
    for (auto input : inputs) {
      Edge edge(input, InputPort(this, idx));
      if (edge.isDataValue()) {
        count++;
      }
    }
    return count;
  }

  SmallVector<Edge> getOutEdges() {
    SmallVector<Edge> result;
    size_t idx = 0;
    for (auto outputs : this->outputs) {
      for (auto output : outputs) {
        result.push_back(Edge(OutputPort(this, idx), output));
      }
      idx++;
    }
    return result;
  }

  size_t getNumOutDataEdges() {
    size_t count = 0;
    size_t idx = 0;
    for (auto output : dataOutputs) {
      if (output) {
        count += outputs[idx].size();
      }
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
    for (auto input : inputs) {
      if (input.getNode() && input.getNode()->isDataValue(input.getIdx())) {
        return true;
      }
    }
    return false;
  }

  std::string getLabel() {
    if (op) {
      return op->getName().getStringRef().str();
    }
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (isa<tt::FuncOp>(parentOp)) {
        return "arg " + std::to_string(blockArg.getArgNumber());
      } else if (isa<scf::ForOp>(parentOp)) {
        if (blockArg.getArgNumber() == 0) {
          return "ind var";
        }
        return "iter arg " + std::to_string(blockArg.getArgNumber() - 1);
      }
      assert(false);
    }
    if (auto result = dyn_cast<OpResult>(value)) {
      return "result " + std::to_string(result.getResultNumber());
    }
    assert(false);
  }

  void setGroup(Group *group) {
    for (auto current_group : groups) {
      current_group->remove(this);
    }
    groups.clear();
    groups.insert(group);
    group->add(this);
  }

  void addGroup(Group *group) {
    groups.insert(group);
    group->add(this);
  }

  void addGroups(const SetVector<Group *> &groups) {
    this->groups.insert(groups.begin(), groups.end());
    for (auto group : groups) {
      group->add(this);
    }
  }

  bool hasGroup() const { return !groups.empty(); }

  Group *getGroup() const {
    assert(groups.size() == 1);
    return *(groups.begin());
  }

  const SetVector<Group *> &getGroups() const { return groups; }

  void dump() { std::cout << "node '" << getLabel() << "'\n"; }

private:
  Node *parent = nullptr;
  Operation *op = nullptr;
  Value value;
  size_t idx = 0;

  SmallVector<std::unique_ptr<Node>> nodes;
  SmallVector<Node *> defines;

  SmallVector<OutputPort> inputs;
  SmallVector<SmallVector<InputPort>> outputs;
  SmallVector<bool> dataOutputs;

  SetVector<Group *> groups;
};

class Graph {
public:
  explicit Graph(Operation *op) : root(new Node(op)) {}

  Node *getRoot() { return root.get(); }

  Group *addGroup() { return groups.emplace_back(new Group).get(); }

  const SmallVector<std::unique_ptr<Group>> &getGroups() const {
    return groups;
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
  SmallVector<std::unique_ptr<Group>> groups;
};

bool isSIMTOp(Operation *op) {
  if (!op->getDialect() || !llvm::isa<arith::ArithDialect>(op->getDialect())) {
    return false;
  }
  auto types = llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes());
  for (Type type : types) {
    if (isa<RankedTensorType>(type)) {
      return true;
    }
  }
  return false;
}

Flags getNodeFlags(Node *node) {
  if (node->isOp()) {
    auto op = node->getOp();
    if (isa<tt::LoadOp, tt::DescriptorLoadOp>(op)) {
      return Flags::LOAD;
    }
    if (isa<tt::StoreOp, tt::DescriptorStoreOp>(op)) {
      return Flags::STORE;
    }
    if (isa<ttng::MMAv5OpInterface>(op)) {
      return Flags::MMA;
    }
    if (isSIMTOp(op)) {
      return Flags::SIMT;
    }
  }
  return Flags::NONE;
}

void Group::add(Node *node) {
  nodes.insert(node);
  flags |= getNodeFlags(node);
}

void Group::merge(Group *lhs, Group *rhs) {
  auto nodes = lhs->getNodes();
  for (auto node : nodes) {
    node->setGroup(rhs);
  }
}

Node *Edge::getFromNode() const { return from.getNode(); }
size_t Edge::getFromIdx() const { return from.getIdx(); }

Node *Edge::getToNode() const { return to.getNode(); }
size_t Edge::getToIdx() const { return to.getIdx(); }

bool Edge::isDataValue() const {
  return from.getNode()->isDataValue(from.getIdx());
}

bool Edge::crossesGroups() const {
  return isDataValue() && from.getNode()->hasGroup() &&
         to.getNode()->hasGroup() &&
         from.getNode()->getGroup() != to.getNode()->getGroup();
}

Type Edge::getType() const {
  auto fromNode = from.getNode();
  if (fromNode->isOp()) {
    return fromNode->getOp()->getResult(from.getIdx()).getType();
  } else {
    return fromNode->getValue().getType();
  }
}

std::unique_ptr<Graph> buildGraph(ModuleOp m) {
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
          for (auto &region : op->getRegions()) {
            for (auto &block : region) {
              for (auto &op : block) {
                visitOperation(node, &op);
              }
            }
          }

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

          for (auto &region : op->getRegions()) {
            for (auto &block : region) {
              for (auto &op : block) {
                visitOperation(node, &op);
              }
            }
          }

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

          for (auto &region : op->getRegions()) {
            for (auto &block : region) {
              for (auto &op : block) {
                visitOperation(node, &op);
              }
            }
          }

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
            // TODO
            assert(false);
          }

        } else if (isa<tt::ReturnOp>(op)) {
          // omit

        } else {
          // assert(op->getNumRegions() == 0);
          auto node =
              graph->addNode(op, op->getNumOperands(), op->getNumResults());
          nodes[op] = node;
          for (size_t idx = 0; idx < op->getNumOperands(); idx++) {
            operands[std::make_pair(op, idx)] = InputPort(node, idx);
          }
          for (const auto &result : op->getResults()) {
            values.push_back(std::make_pair(
                OutputPort(node, result.getResultNumber()), result));
          }
        }
      };

  auto graph = std::make_unique<Graph>(m.getOperation());
  for (auto &block : m.getRegion()) {
    for (auto &op : block) {
      visitOperation(graph->getRoot(), &op);
    }
  }

  for (auto [outputPort, value] : values) {
    for (auto &use : value.getUses()) {
      auto op = use.getOwner();
      if (op) { // && nodes.find(op) != nodes.end()) {
        auto key = std::make_pair(op, use.getOperandNumber());
        if (operands.find(key) != operands.end()) {
          auto inputPort = operands[key];
          // std::cout << "'" << outputPort.getNode()->getLabel() << "' "
          //           << outputPort.getIdx() << " -> '" <<
          //           inputPort.getNode()->getLabel()
          //           << "' " << inputPort.getIdx() << std::endl;
          Node::addEdge(outputPort, inputPort);
        } else {
          assert(false);
          // std::cout << "WARNING: use not found for op "
          //          << op->getName().getStringRef().str()
          //          << " with operand number " << use.getOperandNumber()
          //          << "\n";
        }
      } else {
        assert(false);
        // std::cout << "WARNING: use is not owned by an op\n";
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
      if (isa<tt::LoadOp, tt::DescriptorLoadOp>(op)) {
        node->setDataValue(0);
        values.push_back({node, 0});
      }
      if (isa<ttng::TMEMLoadOp>(op)) {
        node->setDataValue(0);
        values.push_back({node, 0});
        node->setDataValue(1);
        values.push_back({node, 1});
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

void initialGroupAssignment(Graph *graph) {
  graph->walk([&](Node *node) {
    if (node->isData()) {
      auto group = graph->addGroup();
      node->setGroup(group);
    }
  });
}

SmallVector<Edge> getCrossingEdges(Graph *graph) {
  SmallVector<Edge> edges;
  for (auto &group : graph->getGroups()) {
    for (auto node : group->getNodes()) {
      for (auto edge : node->getOutEdges()) {
        if (!edge.crossesGroups()) {
          continue;
        }
        edges.push_back(edge);
      }
    }
  }
  return edges;
}

SmallVector<std::pair<std::string, std::function<bool(Edge)>>> heuristics = {
    // tma load followed by local alloc always in same group
    {"tma_load_local_alloc",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       if (!from->isOp() || !to->isOp()) {
         return false;
       }
       return isa<tt::DescriptorLoadOp>(from->getOp()) &&
              isa<ttg::LocalAllocOp>(to->getOp());
     }},

    // memdesc trans followed by mma always in same group
    {"trans_mma",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       if (!from->isOp() || !to->isOp()) {
         return false;
       }
       return isa<ttg::MemDescTransOp>(from->getOp()) &&
              isa<ttng::MMAv5OpInterface>(to->getOp());
     }},

    // straight sequence of SIMT/NONE ops merges together
    {"simt_sequence",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       if (from->getNumOutDataEdges() > 1 || to->getNumInDataEdges() > 1) {
         return false;
       }
       return (from->getGroup()->getFlags() == Flags::NONE ||
               from->getGroup()->getFlags() == Flags::SIMT) &&
              (to->getGroup()->getFlags() == Flags::NONE ||
               to->getGroup()->getFlags() == Flags::SIMT);
     }},

    // straight sequence of NONE ops merges together
    {"none_sequence",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       if (from->getNumOutDataEdges() > 1 || to->getNumInDataEdges() > 1) {
         return false;
       }
       return from->getGroup()->getFlags() == Flags::NONE &&
              to->getGroup()->getFlags() == Flags::NONE;
     }},

    // NONE ops preceeding MMA merged together
    {"mma_none",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return from->getGroup()->getFlags() == Flags::NONE &&
              to->getGroup()->getFlags() == Flags::MMA;
     }},

    // NONE ops following LOAD merge together
    {"load_none",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return from->getGroup()->getFlags() == Flags::LOAD &&
              to->getGroup()->getFlags() == Flags::NONE;
     }},

    // NONE ops preceeding STORE merge together
    {"store_none",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return from->getGroup()->getFlags() == Flags::NONE &&
              to->getGroup()->getFlags() == Flags::STORE;
     }},

    // NONE ops preceeding SIMT merge together
    {"simt_none",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return from->getGroup()->getFlags() == Flags::NONE &&
              to->getGroup()->getFlags() & Flags::SIMT;
     }},

    // SIMT ops preceeding STORE merge together
    {"store_simt",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return from->getGroup()->getFlags() == Flags::SIMT &&
              to->getGroup()->getFlags() & Flags::STORE;
     }},
};

SmallVector<std::pair<std::string, std::function<bool(Edge)>>> constraints = {
    // don't merge tmem load into mma group
    {"tmem_load",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return !((from->getGroup()->getFlags() == Flags::MMA && to->isOp() &&
                 isa<ttng::TMEMLoadOp>(to->getOp())) ||
                (to->getGroup()->getFlags() == Flags::MMA && from->isOp() &&
                 isa<ttng::TMEMLoadOp>(from->getOp())));
     }},
    // don't merge tmem store into mma group
    {"tmem_store",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return !((to->getGroup()->getFlags() == Flags::MMA && from->isOp() &&
                 isa<ttng::TMEMStoreOp>(from->getOp())) ||
                (from->getGroup()->getFlags() == Flags::MMA && to->isOp() &&
                 isa<ttng::TMEMStoreOp>(to->getOp())));
     }},
};

void mergeGroups(Graph *graph) {
  // Note: this implementation is slow. It can be improved by incrementally
  // updating the data structures rather than rebuilding the whole lot when a
  // rule is applied
  int iter = 0;
  bool changed = false;
  do {
    changed = false;

    auto crossingEdges = getCrossingEdges(graph);

    for (auto [name, apply] : heuristics) {
      for (auto edge : crossingEdges) {
        if (apply(edge)) {

          // check if applying the heuristic will observe the contraints
          bool ok = true;
          for (auto [name, constraint] : constraints) {
            if (!constraint(edge)) {
              ok = false;
            }
          }
          if (ok) {
            // std::cout << "apply " << name << "\n";
            // std::cout << edge.getFromNode()->getLabel() << " -> "
            //           << edge.getToNode()->getLabel() << "\n";
            // std::cout << "merge groups " << edge.getFromNode()->getGroup()
            //           << ", " << edge.getToNode()->getGroup() << "\n";

            // merge the groups
            auto from_group = edge.getFromNode()->getGroup();
            auto to_group = edge.getToNode()->getGroup();
            Group::merge(from_group, to_group);

            changed = true;
            break;
          }
        }
      }
      if (changed) {
        break;
      }
    }

    iter++;
  } while (changed && iter < 10000);
  // std::cout << "iter = " << iter << "\n";

  // merge all load groups
  {
    llvm::SmallVector<Group *> load_groups;
    for (auto &group : graph->getGroups()) {
      if (group->getFlags() & Flags::LOAD) {
        load_groups.push_back(group.get());
      }
    }
    std::cout << "load group count = " << load_groups.size() << "\n";
    if (load_groups.size() > 1) {
      auto load_group = load_groups.front();
      for (auto it = load_groups.begin() + 1; it != load_groups.end(); it++) {
        Group::merge(*it, load_group);
      }
    }
  }
}

void propagateGroups(Graph *graph) {
  // propagate groups to parent ops
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
    if (is_leaf) {
      leaves.push_back(node);
    }
  });

  bool changed = true;
  while (changed) {
    for (auto leaf : leaves) {
      // groups for leaf are union of groups of all ops contained in the leaf
      SetVector<Group *> groups;
      for (auto &node : leaf->getNodes()) {
        groups.insert(node->getGroups().begin(), node->getGroups().end());
      }
      leaf->addGroups(groups);

      // propagate to parent nodes
      auto node = leaf->getParent();
      while (node) {
        // include union of groups of ops in the parent
        for (auto &child : node->getNodes()) {
          groups.insert(child->getGroups().begin(), child->getGroups().end());
        }
        node->addGroups(groups);
        node = node->getParent();
      }
    }

    // propagate groups to non-data nodes
    {
      SmallVector<Node *> nodes;
      // include nodes with regions
      graph->walk([&](Node *node) {
        if (!node->getNodes().empty()) {
          nodes.push_back(node);
        }
      });
      // include data nodes
      for (auto &group : graph->getGroups()) {
        for (auto &node : group->getNodes()) {
          if (node->isData()) {
            nodes.push_back(node);
          }
        }
      }

      changed = false;
      for (auto node : nodes) {
        SmallVector<Node *> stack;
        DenseSet<Node *> seen;
        auto groups = node->getGroups();
        stack.push_back(node);
        seen.insert(node);

        while (!stack.empty()) {
          auto node = stack.back();
          stack.pop_back();

          for (auto edge : node->getInEdges()) {
            if (edge.isDataValue()) {
              continue;
            }
            auto fromNode = edge.getFromNode();
            auto numGroupsBefore = fromNode->getGroups().size();
            fromNode->addGroups(groups);
            auto numGroupsAfter = fromNode->getGroups().size();
            changed |= (numGroupsBefore != numGroupsAfter);

            if (seen.count(edge.getFromNode()) == 0) {
              stack.push_back(fromNode);
              seen.insert(fromNode);
            }
          }
        }
      }
    }
  }
}

void visualize(std::string path, Graph *graph) {
  std::ofstream dot(path);
  dot << "digraph G {\n";

  DenseMap<Node *, size_t> node_ids;
  DenseMap<Group *, size_t> group_ids;
  DenseMap<Group *, std::string> group_colors;

  auto getGroupId = [&](Group *group) {
    if (group_ids.count(group) == 0) {
      group_ids[group] = group_ids.size();
    }
    return group_ids[group];
  };

  auto getGroupColor = [&](Group *group) {
    if (group_colors.count(group) == 0) {
      size_t color = group_colors.size() + 1;
      color = (color % 12) + 1;
      group_colors[group] = std::string("/set312/") + std::to_string(color);
    }
    return group_colors[group];
  };

  // add nodes
  std::function<void(Node *)> visitNodes = [&](Node *graph) {
    for (auto &node_obj : graph->getNodes()) {
      auto node = node_obj.get();
      node_ids[node] = node_ids.size();

      // if (node->isOp()) {
      if (!node->getNodes().empty()) {
        dot << "subgraph cluster_cx" << node_ids[node] << " {\n";
      }
      dot << "x" << node_ids[node] << "[shape=plaintext, ";
      if (node->isData()) {
        dot << "color=blue, ";
      }
      dot << "label=<";
      dot << "<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\">";
      if (node->getNumInputs() > 1) {
        dot << "<TR>";
        for (size_t idx = 0; idx < node->getNumInputs(); idx++) {
          dot << "<TD PORT=\"in" << idx << "\">" << idx << "</TD>";
        }
        dot << "</TR>";
      }
      dot << "<TR><TD PORT=\"inout\"";
      size_t colspan = std::max(node->getNumInputs(), node->getNumOutputs());
      if (colspan > 0) {
        dot << " COLSPAN=\"" << colspan << "\"";
      }
      dot << ">";

      dot << "<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\"><TR>";
      if (node->hasGroup()) {
        for (auto group : node->getGroups()) {
          dot << "<TD BGCOLOR=\"" << getGroupColor(group) << "\">"
              << getGroupId(group) << "</TD>";
        }
      }
      dot << "<TD>" << node->getLabel() << "</TD></TR></TABLE>";

      dot << "</TD></TR>";
      if (node->getNumOutputs() > 1) {
        dot << "<TR>";
        for (size_t idx = 0; idx < node->getNumOutputs(); idx++) {
          dot << "<TD PORT=\"out" << idx << "\">" << idx << "</TD>";
        }
        dot << "</TR>";
      }
      dot << "</TABLE>>];\n";
      if (!node->getNodes().empty()) {
        visitNodes(node);
        dot << "}\n";
      }
      // } else {
      //   assert(node->getNodes().empty());
      //   assert(node->getNumInputs() == 1);
      //   assert(node->getNumOutputs() == 1);
      //   dot << "x" << node_ids[node] << "[shape=ellipse, ";
      //   if (node->isData()) {
      //     dot << "color=blue, ";
      //   }
      //   dot << "label=\"";
      //   dot << node->getLabel();
      //   dot << "\"];\n";
      // }
    }
  };
  visitNodes(graph->getRoot());

  // add edges
  std::function<void(Node *)> visitEdges = [&](Node *node) {
    // for (auto definedNode : node->getDefines()) {
    //   dot << "x" << node_ids[node];
    //   if (node->isOp()) {
    //     dot << ":inout";
    //   }
    //   dot << " -> x" << node_ids[definedNode];
    //   if (definedNode->isOp()) {
    //     dot << ":inout";
    //   }
    //   dot << " [dir=none, style=dashed];\n";
    // }
    size_t idx = 0;
    for (auto inputPorts : node->getOutputs()) {
      OutputPort outputPort{node, idx};
      for (auto inputPort : inputPorts) {
        Edge edge(outputPort, inputPort);
        dot << "x" << node_ids[outputPort.getNode()];
        // if (outputPort.getNode()->isOp()) {
        dot << ":";
        if (outputPort.getNode()->getNumOutputs() == 1) {
          dot << "inout";
        } else {
          dot << "out" << outputPort.getIdx();
        }
        //}
        dot << " -> ";
        dot << "x" << node_ids[inputPort.getNode()];
        // if (inputPort.getNode()->isOp()) {
        dot << ":";
        if (inputPort.getNode()->getNumInputs() == 1) {
          dot << "inout";
        } else {
          dot << "in" << inputPort.getIdx();
        }
        //}
        if (edge.isDataValue()) {
          if (edge.crossesGroups()) {
            dot << "[color=\"red\"]";
          } else {
            dot << "[color=\"blue\"]";
          }
        }
        dot << ";\n";
      }
      idx++;
    }
    for (auto &node : node->getNodes()) {
      visitEdges(node.get());
    }
  };
  visitEdges(graph->getRoot());

  dot << "}\n";
}

void serialize(Graph *graph) {
  Builder b(graph->getRoot()->getOp()->getContext());

  DenseMap<Group *, size_t> group_ids;

  auto getGroupId = [&](Group *group) {
    if (group_ids.count(group) == 0) {
      group_ids[group] = group_ids.size();
    }
    return group_ids[group];
  };

  auto setGroupsAttr = [&](Operation *op, const std::string &attrName,
                           Node *node) {
    SmallVector<int> group_ids;
    for (auto group : node->getGroups()) {
      group_ids.push_back(getGroupId(group));
    }
    std::sort(group_ids.begin(), group_ids.end());
    SmallVector<Attribute, 4> groups;
    for (auto id : group_ids) {
      groups.push_back(b.getI32IntegerAttr(id));
    }
    op->setAttr(attrName, ArrayAttr::get(op->getContext(), groups));
  };

  graph->walk([&](Node *node) {
    if (node->isOp()) {
      setGroupsAttr(node->getOp(), "ttg.partitions", node);
    } else {
      auto value = node->getValue();
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        auto parentOp = blockArg.getOwner()->getParentOp();
        if (isa<tt::FuncOp>(parentOp)) {
          // nothing for func ops
        } else if (isa<scf::ForOp>(parentOp)) {
          if (blockArg.getArgNumber() == 0) {
            // nothing for induction variable
          } else {
            // for op iter args
            setGroupsAttr(parentOp,
                          "ttg.partitions." +
                              std::to_string(blockArg.getArgNumber() - 1),
                          node);
          }
        } else {
          assert(false);
        }
      } else if (auto result = dyn_cast<OpResult>(value)) {
        auto op = result.getOwner();
        if (isa<scf::ForOp>(op)) {
          // do nothing (handled by block arg)
        } else if (isa<scf::IfOp>(op)) {
          // result of an if
          setGroupsAttr(
              op, "ttg.partitions." + std::to_string(result.getResultNumber()),
              node);
        } else {
          assert(false);
        }
      } else {
        assert(false);
      }
    }
  });
}

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUPARTITIONANALYSIS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct PartitionAnalysis
    : public triton::gpu::impl::TritonGPUPartitionAnalysisBase<
          PartitionAnalysis> {
  using TritonGPUPartitionAnalysisBase::TritonGPUPartitionAnalysisBase;

  void runOnOperation() override;
};
} // namespace

void PartitionAnalysis::runOnOperation() {
  ModuleOp m = getOperation();
  auto func = cast<tt::FuncOp>(m.getRegion().front().front());

  bool dump = tools::getBoolEnv("PARTITION_ANALYSIS_ENABLE_DUMP");

  auto graph = buildGraph(m);

  auto initValues = initialDataValues(graph.get());
  propagateDataValues(initValues);
  if (dump) {
    visualize(std::string("graph-input-") + func.getSymName().str() + ".dot",
              graph.get());
  }
  initialGroupAssignment(graph.get());
  if (dump) {
    visualize(std::string("graph-initial-") + func.getSymName().str() + ".dot",
              graph.get());
  }
  mergeGroups(graph.get());
  if (dump) {
    visualize(std::string("graph-merged-") + func.getSymName().str() + ".dot",
              graph.get());
  }
  propagateGroups(graph.get());
  if (dump) {
    visualize(std::string("graph-final-") + func.getSymName().str() + ".dot",
              graph.get());
  }
  serialize(graph.get());
}
