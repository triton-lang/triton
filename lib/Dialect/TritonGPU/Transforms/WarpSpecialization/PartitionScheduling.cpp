#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionSchedulingUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton::gpu {

// This pass assigns partitions to ops within each warp specialized loop.
//
// Ops are first categorized as either "data" ops (which operate on tiles of
// data, for example load/store/mma ops) or "non-data" ops (for example index
// calculations).
//
// A dataflow graph representation of the program is constructed: every edge in
// the graph represents an MLIR value, and every node represents an MLIR
// operation or block argument.
//
// Initially all nodes for "data" ops are assigned to a new partition. A set of
// heuristics is then applied to every edge that crosses partitions (connects a
// pair of nodes assigned to different partitions). When a heuristic matches,
// the two partitions are merged into a single partition. This is done up until
// a fixed point is reached. A second set of heuristics is run on every
// pair of partitions, merging them until a fixed point is reached.
//
// After the heuristics have been applied, all data ops are assigned to a
// single partition. These partition assignments are then propagated to all
// "non-data" ops. This pulls all of the necessary index calculations etc. into
// the partitions that require them (possibly multiple).
//
// Finally the partition assignments in the dataflow graph are serialized to
// attributes, and the temporary data structure is discarded.

#define GEN_PASS_DEF_TRITONGPUPARTITIONSCHEDULING
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritongpu-partition-scheduling"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton;
using namespace partition_scheduling_detail;

namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

using Partition = partition_scheduling_detail::Partition; // resolve ambiguity

template <typename... Args> bool node_isa(Node *node) {
  return node->isOp() && isa<Args...>(node->getOp());
}

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
  visitOperation(graph->getRoot(), region);

  for (auto [outputPort, value] : values) {
    for (auto &use : value.getUses()) {
      auto op = use.getOwner();
      auto key = std::make_pair(op, use.getOperandNumber());
      if (operands.find(key) != operands.end()) {
        auto inputPort = operands[key];
        Node::addEdge(outputPort, inputPort);
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
      if (isa<ttng::TMEMLoadOp>(op)) {
        node->setDataValue(0);
        values.push_back({node, 0});
        node->setDataValue(1);
        values.push_back({node, 1});
      }
      if (isa<nvidia_gpu::TCGen5MMAOp>(op)) {
        node->setDataValue(0);
        values.push_back({node, 0});
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

void deserializeManualPartitions(Operation *region, Graph *graph) {
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
       assert(out_edges.size() == 1);
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

    // TMEM load merges with consumer
    // FIXME: limit to single consumer?
    {"tmem_load",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return node_isa<ttng::TMEMLoadOp>(from);
     }},

    // TMEM and STORE groups merge
    {"tmem_store",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return isTMEM(from) && isStore(to);
     }},

    // NONE/cheap SFU merges with consumer (except LOAD, MMA or costly SFU)
    {"none_consumer",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return (isNone(from) || (isSFU(from) && !isCostlySFU(from))) &&
              !isNone(to) && !isMMA(to) && !isLoad(to) && !isCostlySFU(to);
     }},

    // NONE merges with costly producer (except LOAD or MMA)
    // This will prefer to merge NONE nodes into costly groups, rather than
    // non-costly groups
    // e.g. in the two SFU groups of attention kernels
    {"none_producer_costly",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return isNone(to) && !isNone(from) && !isMMA(from) && !isLoad(from) &&
              from->getPartition()->getCost() > 256;
     }},

    // NONE merges with producer (except LOAD or MMA)
    {"none_producer",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return isNone(to) && !isNone(from) && !isMMA(from) && !isLoad(from);
     }},

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

    // merge connected NONE and MANUAL partitions together
    {"connected_none_manual",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return (isOnlyNone(from) && isManual(to)) ||
              (isOnlyNone(to) && isManual(from));
     }},

    // merge connected partitions together if edge between is expensive
    // TODO: this might be better expressed as a horizontal rule,
    // that aims to keep shmem usage under the limit
    {"connected",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       return !isLoad(from) && !isLoad(to) && !isMMA(from) && !isMMA(to) &&
              edge.getSize() > 16384; // FIXME: seemingly arbitrary size...
     }},

    // store group not used by an mma/dot op should be merged
    {"load_epilog",
     [](Edge edge) {
       auto from = edge.getFromNode();
       auto to = edge.getToNode();
       if (!isLoad(from))
         return false;

       SmallVector<Node *> stack;
       DenseSet<Node *> seen;
       stack.push_back(from);
       seen.insert(from);

       while (!stack.empty()) {
         auto node = stack.back();
         stack.pop_back();
         if (isMMA(node) || (node->isOp() && isa<tt::DotOp>(node->getOp()))) {
           return false;
         } else {
           for (auto edge : node->getOutEdges()) {
             if (!seen.contains(edge.getToNode())) {
               stack.push_back(edge.getToNode());
               seen.insert(edge.getToNode());
             }
           }
         }
       }

       return true;
     }},
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
};

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

        // merge TMEM partitions together, if they use the same tmem alloc
        // aref does not support tmem with more than 2 partitions
        // and the tmem_alloc'd memory can maximally be used by an MMA
        // partition and a TMEM partition
        {"tmem",
         [](Partition *a, Partition *b) {
           auto a_is_tmem = (a->getFlags() & Flags::TMEM);
           auto b_is_tmem = (b->getFlags() & Flags::TMEM);
           if (!a_is_tmem || !b_is_tmem)
             return false;
           auto allocs_a = getTMEMAllocs(a);
           auto allocs_b = getTMEMAllocs(b);
           // if the sets are overlapping, alloc is used by both TMEM partitions
           for (auto alloc_a : allocs_a)
             if (allocs_b.contains(alloc_a))
               return true;
           return false;
         }},
};

void mergePartitions(Graph *graph, std::string funcName,
                     VisualizationInfo &vis_info) {
  LLVM_DEBUG({ llvm::errs() << "#### applying heuristics...\n"; });

  // initial worklist is list of all edges that cross partitions
  auto crossingEdges = getCrossingEdges(graph);
  bool changed = false;
  do {
    changed = false;
    LLVM_DEBUG({
      llvm::errs() << "\n"
                   << crossingEdges.size() << " crossing edges remaining\n";
    });

    for (auto [name, apply] : heuristics) {
      for (auto it = crossingEdges.begin(); it != crossingEdges.end();) {
        auto edge = *it;

        // remove edges that no longer cross partitions from the worklist
        if (!edge.crossesPartitions()) {
          it = crossingEdges.erase(it);
          continue;
        }

        if (apply(edge)) {
          // check if applying the heuristic will satisfy the constraints
          bool ok = true;
          for (auto [name, constraint] : constraints) {
            if (!constraint(edge)) {
              ok = false;
              break;
            }
          }
          if (!ok) {
            it++;
            continue;
          }

          LLVM_DEBUG({
            llvm::dbgs() << "\napply heuristic \"" << name << "\"\n";
            llvm::dbgs() << edge.getFromNode()->getLabel() << " -> "
                         << edge.getToNode()->getLabel() << "\n";
            llvm::dbgs() << "partitions " << edge.getFromNode()->getPartition()
                         << " -> " << edge.getToNode()->getPartition() << "\n";
            llvm::dbgs() << "flags "
                         << edge.getFromNode()->getPartition()->getFlags()
                         << " -> "
                         << edge.getToNode()->getPartition()->getFlags()
                         << "\n";
          });

          // merge the partitions
          auto from_partition = edge.getFromNode()->getPartition();
          auto to_partition = edge.getToNode()->getPartition();
          Partition::merge(from_partition, to_partition);

          visualize(funcName, "merge-step", std::string("merge: rule ") + name,
                    graph, vis_info);
          crossingEdges.erase(it);

          changed = true;
          break;
        }

        it++;
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
      for (auto partition : graph->getPartitions())
        all_partitions.push_back(partition);
      for (auto [name, apply] : partition_heuristics) {
        for (auto partitionA : all_partitions) {
          for (auto partitionB : all_partitions) {
            if (partitionA == partitionB)
              continue;
            if (apply(partitionA, partitionB)) {
              LLVM_DEBUG({
                llvm::errs() << "\nmerge \"" << name << "\" ----\n";
                partitionA->dump();
                partitionB->dump();
              });
              Partition::merge(partitionA, partitionB);
              visualize(funcName, "merge-step",
                        std::string("merge: rule ") + name, graph, vis_info);
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

  LLVM_DEBUG({ llvm::errs() << "\n#### heuristics done\n"; });
}

void propagatePartitions(Graph *graph, std::string funcName,
                         VisualizationInfo &vis_info) {
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
        }
      }
    }
  }

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

  visualize(funcName, "propagate", "propagate forward", graph, vis_info);

  // propagate partitions of tt.reduce into its body
  graph->walk([&](Node *node) {
    if (node->isOp() && isa<tt::ReduceOp>(node->getOp())) {
      auto partitions = node->getPartitions();
      node->walk(
          [&](Node *child_node) { child_node->addPartitions(partitions); });
    }
  });

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
  visualize(funcName, "duplicate", "before duplicate cheap ops", graph,
            vis_info);

  // for each partition:
  // look at all crossing edges leaving the partition
  // do a depth first search through NONE nodes, if we hit the same partition
  // assign all nodes on that path to the partition
  for (auto partition : graph->getPartitions()) {

    auto crossingEdges = getOutCrossingEdges(partition);

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
        return (getNodeFlags(node) == Flags::NONE ||
                getNodeFlags(node) == Flags::SFU);
      };

      if (!isCandidate(edge.getToNode()))
        continue;

      auto update = [&]() {
        std::map<Node *, Node *> parentMap;

        SmallVector<Node *> stack;
        stack.push_back(start);
        DenseSet<Node *> seen;

        while (!stack.empty()) {
          auto node = stack.back();
          stack.pop_back();
          if (!seen.contains(node)) {
            seen.insert(node);
            for (auto edge : node->getOutEdges()) {
              auto child = edge.getToNode();
              if (!seen.contains(child)) {
                if (child->getPartitions().size() != 1 || !isCandidate(child)) {
                  // do nothing
                } else if (child->getPartition() == partition) {
                  parentMap.emplace(child, node);
                  stack.push_back(child);
                } else if (child->getPartition() == startPartition) {
                  // found a path, set all nodes on the path to the partition
                  node->addPartition(startPartition);
                  while (parentMap.find(node) != parentMap.end()) {
                    node = parentMap[node];
                    node->addPartition(startPartition);
                  }

                  visualize(funcName, "duplicate", "duplicate cheap ops", graph,
                            vis_info);

                  return;
                }
              }
            }
          }
        }
      };
      update();
    }
  }

  visualize(funcName, "duplicate", "duplicate cheap ops done", graph, vis_info);
}

void serialize(size_t idx, Operation *region, Graph *graph) {

  SetVector<Operation *> alreadyWritten;

  auto context = graph->getRoot()->getOp()->getContext();
  Builder b(context);

  // annotate loop with index
  region->setAttr(kWarpSpecializeTagAttrName, b.getI32IntegerAttr(idx));

  auto setPartitionsAttr = [&](Operation *op, Node *node) {
    // not for func op
    if (isa<tt::FuncOp>(op))
      return;

    // Note: we may have multiple nodes per op, so we merge the partition
    // ids for all nodes of the op
    SetVector<int> partitionIds;
    if (alreadyWritten.contains(op)) {
      // if we already serialized a node to this op, merge those partition ids
      // with the node being serialized
      partitionIds = getPartitionIds(op);
    }
    alreadyWritten.insert(op);
    for (auto partition : node->getPartitions())
      partitionIds.insert(*partition->id);
    auto partitionIdsList = partitionIds.takeVector();
    std::sort(partitionIdsList.begin(), partitionIdsList.end());
    auto partitionsAttr = b.getDenseI32ArrayAttr(partitionIdsList);
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
    auto id = *partition->id;
    while (id >= stages.size())
      stages.push_back(b.getI32IntegerAttr(0));
    stages[id] = b.getI32IntegerAttr(partition->getStage());
  }
  region->setAttr(kPartitionStagesAttrName, b.getArrayAttr(stages));
}

void duplicateViewOps(Graph *graph) {
  // Ensure all view ops (e.g. broadcast/expand dims) have a single user,
  // by duplicating nodes where necessary

  SmallVector<Node *> viewOps;

  graph->walk([&](Node *node) {
    if (node->isData() && node->isOp() && isViewOp(node->getOp()))
      viewOps.push_back(node);
  });

  while (!viewOps.empty()) {
    auto node = viewOps.pop_back_val();
    auto op = node->getOp();

    assert(op->getResults().size() == 1);

    auto outEdges = node->getOutEdges();

    bool first = true;
    for (auto edge : outEdges) {
      if (!first) {
        auto newNode = node->getParent()->addNode(op, op->getNumOperands(),
                                                  op->getNumResults());

        // remove old edge
        Node::removeEdge(edge);

        // add new edge
        OutputPort outputPort(newNode, 0);
        OutputPort inputPort(edge.getToNode(), edge.getToIdx());
        Node::addEdge(outputPort, inputPort);

        // add operands of new node
        for (auto inEdge : node->getInEdges()) {
          Node::addEdge(inEdge.getFrom(),
                        InputPort(newNode, inEdge.getToIdx()));
        }

        // copy data values
        for (auto idx = 0; idx < op->getNumResults(); idx++) {
          if (node->isDataValue(idx)) {
            newNode->setDataValue(idx);
          }
        }
      }
      first = false;
    }
  }
}

void assignPartitionIds(Graph *graph) {
  size_t idx = 0;

  SmallVector<Partition *> store_partitions;
  SmallVector<Partition *> mma_partitions;
  SmallVector<Partition *> load_partitions;
  SmallVector<Partition *> other_partitions;

  for (auto partition : graph->getPartitions()) {
    if (partition->getFlags() & Flags::STORE)
      store_partitions.push_back(partition);
    else if (partition->getFlags() & Flags::MMA)
      mma_partitions.push_back(partition);
    else if (partition->getFlags() & Flags::LOAD)
      load_partitions.push_back(partition);
    else
      other_partitions.push_back(partition);
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

void assignPartitionsForOpsWithNoUse(Graph *graph) {
  // nodes with no partition placed in same partition as other ops in the
  // region or default partition if none. Note: we can't just use partitions
  // of parent op, as this includes things like tmem tokens
  Partition *defaultPartition = nullptr;
  for (auto partition : graph->getPartitions())
    if (partition->id && *partition->id == 0)
      defaultPartition = partition;
  graph->walk([&](Node *node) {
    if (node->getPartitions().empty()) {
      bool done = false;
      auto parent = node->getParent();
      if (parent && parent->isOp()) {
        for (auto &otherNode : parent->getNodes()) {
          if (node == otherNode.get())
            continue;
          if (otherNode->isOp() && otherNode->hasPartition()) {
            node->addPartitions(otherNode->getPartitions());
            done = true;
          }
        }
      }
      if (!done) {
        if (defaultPartition == nullptr) {
          // default partition doesn't exist, create one
          defaultPartition = graph->addPartition();
          defaultPartition->id = 0;
        }
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
    // find ops to partition
    SmallVector<Operation *> ops;
    getOperation().walk([&](scf::ForOp op) {
      if (op->hasAttr(kWarpSpecializeAttrName))
        ops.push_back(op);
    });

    // run partitioner on each op
    size_t idx = 0;
    for (auto op : ops) {
      analyze(idx, op);
      cloneMultiPartitionDataOps(op);
      idx++;
    }
  }

private:
  void analyze(size_t idx, Operation *op) {
    using namespace partition_scheduling_detail;

    auto func = op->getParentOfType<FuncOp>();

    VisualizationInfo vis_info;
    auto key = func.getSymName().str() + "_" + std::to_string(idx);

    auto graph = buildGraph(op);
    visualize(key, "input", "input", graph.get(), vis_info);
    auto initValues = initialDataValues(graph.get());
    propagateDataValues(initValues);
    visualize(key, "input", "after data values", graph.get(), vis_info);
    duplicateViewOps(graph.get());
    visualize(key, "input", "after duplicate view ops", graph.get(), vis_info);
    deserializeManualPartitions(op, graph.get());
    visualize(key, "input", "final", graph.get(), vis_info);

    initialPartitionAssignment(graph.get());
    visualize(key, "initial", "initial partitions", graph.get(), vis_info);
    mergePartitions(graph.get(), key, vis_info);
    visualize(key, "merge", "merged", graph.get(), vis_info);
    propagatePartitions(graph.get(), key, vis_info);
    visualize(key, "propagate", "propagated", graph.get(), vis_info);

    assignPartitionIds(graph.get());
    visualize(key, "assign-partition-ids", "assign partition ids", graph.get(),
              vis_info);
    // Handle case where ops with no uses (like llvm.intr.assume) get no
    // partition assigned
    assignPartitionsForOpsWithNoUse(graph.get());
    visualize(key, "assign-no-use", "assign no use", graph.get(), vis_info);
    propagatePartitions(graph.get(), key, vis_info);
    visualize(key, "propagate", "propagated", graph.get(), vis_info);
    // Optimization: looks for paths of NONE ops with low cost, from one
    // partition, through another partition, and back to the same partition.
    // Duplicates these to avoid the aref involved (i.e. assign to both
    // partitions)
    duplicateCheapOps(graph.get(), key, vis_info);
    visualize(key, "final", "final", graph.get(), vis_info);

    LLVM_DEBUG({
      llvm::errs() << "\nfinal partitions:\n";
      for (auto &partition : graph->getPartitions())
        partition->dump();
    });

    serialize(idx, op, graph.get());
  }

  void cloneMultiPartitionDataOps(Operation *region) {
    // FIXME: this transformation runs after the partition scheduling is
    // complete It clones "data" ops with multiple partitions assigned, as
    // insert-aref pass cannot currently handly these. E.g. an op assigned to
    // partitions 0,1 will be cloned into two ops, one in partition 0 and the
    // other in partition 1 and all uses are updated correctly.

    using namespace partition_scheduling_detail;

    // build data flow graph to find all data ops
    DenseSet<Operation *> dataOps;
    {
      auto graph = buildGraph(region);
      auto initValues = initialDataValues(graph.get());
      propagateDataValues(initValues);
      graph->walk([&](Node *node) {
        if (node->isOp() && node->isData())
          dataOps.insert(node->getOp());
      });
    }

    // for each partition, find all data ops that are in that partition,
    // and in another partition
    for (auto partition : getPartitionIds(region)) {
      SetVector<int> partitionSet;
      partitionSet.insert(partition);

      SmallVector<Operation *> ops;
      region->walk([&](Operation *op) {
        auto partitions = getPartitionIds(op);
        if (partitions.contains(partition) && partitions.size() > 1 &&
            dataOps.contains(op))
          ops.push_back(op);
      });

      SmallVector<Operation *> oldOps;
      SetVector<Operation *> newOps;
      DenseMap<Operation *, Operation *> mapping;
      for (auto op : ops) {
        auto newOp = OpBuilder(op).clone(*op);
        setPartition(newOp, partitionSet);
        oldOps.push_back(op);
        newOps.insert(newOp);
        mapping[newOp] = op;
        mapping[op] = newOp;
      }

      // rewrite operands
      // if op that produces operand of new op is has a duplicated op,
      // rewrite the operand to use that op
      for (auto newOp : newOps) {
        for (auto &operand : newOp->getOpOperands()) {
          auto value = operand.get();
          if (isa<OpResult>(value)) {
            auto result = cast<OpResult>(value);
            auto producerOp = result.getOwner();
            if (mapping.contains(producerOp)) {
              auto newProducerOp = mapping[producerOp];
              auto newValue =
                  newProducerOp->getResult(result.getResultNumber());
              auto idx = operand.getOperandNumber();
              newOp->setOperand(idx, newValue);
            }
          }
        }
      }

      // rewrite results
      for (auto newOp : newOps) {
        auto oldOp = mapping[newOp];
        for (auto &use : oldOp->getUses()) {
          auto user = use.getOwner();
          assert(user);
          auto userPartitions = getPartitionIds(user);
          // skip if use is not in same partition as new op
          if (userPartitions != partitionSet)
            continue;
          // update the use to use the new op
          auto result = cast<OpResult>(use.get());
          auto idx = result.getResultNumber();
          use.set(newOp->getResult(idx));
        }
      }

      // remove dead code
      bool done = false;
      while (!done) {
        done = true;
        auto op = oldOps.begin();
        for (; op != oldOps.end(); op++) {
          if ((*op)->getUses().empty()) {
            (*op)->erase();
            oldOps.erase(op);
            done = false;
            break;
          }
        }
      }
    }
  }
};

} // namespace mlir::triton::gpu
