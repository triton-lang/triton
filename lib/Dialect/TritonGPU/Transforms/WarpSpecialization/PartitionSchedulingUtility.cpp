#include "triton/Dialect/TritonGPU/Transforms/PartitionSchedulingUtility.h"
#include "mlir/Support/LLVM.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include <iomanip>
#include <sstream>

namespace mlir::triton::gpu::partition_scheduling_detail {

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

Flags getNodeFlags(Node *node) {
  if (node->isOp()) {
    auto op = node->getOp();

    // if it is manually tagged with a node type
    if (op->hasAttr("store"))
      return Flags::STORE;

    if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
      return Flags::LOAD;
    if (isa<tt::DescriptorStoreOp, tt::DescriptorScatterOp>(op))
      return Flags::STORE;
    if (isa<ttng::MMAv5OpInterface>(op) || op->hasAttr("mma"))
      return Flags::MMA;
    if (isa<ttng::TMEMLoadOp, ttng::TMEMStoreOp>(op))
      return Flags::TMEM;
    if (isa<math::Exp2Op>(op))
      return Flags::SFU;
    if (isViewOp(op))
      return Flags::VIEW;
  }
  return Flags::NONE;
}

size_t computeCost(Operation *op) {
  if (auto mma = dyn_cast<nvidia_gpu::TCGen5MMAOp>(op)) {
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
  }

  return 0;
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
  assert(lhs != rhs);

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

  // remove the now empty partition
  lhs->graph->erasePartition(lhs);
}

void Partition::dump() const {
  llvm::errs() << "Partition@" << this << " {\n"
               << "  id=" << id << "\n"
               << "  size=" << nodes.size() << "\n"
               << "  cost=" << cost << "\n"
               << "  flags=" << flags << "\n"
               << "}\n";
}

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
  // FIXME: only considers edges between nodes assigned to single partitions
  // as crossing a boundary
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

void visualize(std::string key, std::string filename, std::string title,
               Graph *graph, VisualizationInfo &info) {

  if (!tools::getBoolEnv("TRITON_PARTITION_SCHEDULING_ENABLE_DUMP_DOT"))
    return;

  const auto dump_data_only =
      tools::getBoolEnv("TRITON_PARTITION_SCHEDULING_DUMP_DATA_ONLY");
  const auto dump_loop_only =
      tools::getBoolEnv("TRITON_PARTITION_SCHEDULING_DUMP_LOOP_ONLY");

  static std::map<std::string, int> keys;
  if (keys.find(key) == keys.end()) {
    keys[key] = 0;
  }
  auto idx = keys[key];
  keys[key]++;

  std::stringstream path;
  path << "graph-" << key << "-" << std::setfill('0') << std::setw(4) << idx
       << "-" << filename << ".dot";

  std::error_code err;
  llvm::raw_fd_ostream dot(path.str(), err);
  assert(!err);

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

  // add nodes
  std::function<void(Node *)> visitNodes = [&](Node *graph) {
    for (auto &node_obj : graph->getNodes()) {
      auto node = node_obj.get();

      if (dump_data_only && !node->isData() && !node->containsData())
        // skip if dumping data nodes only, and this op is non-data or doesn't
        // contain a data node
        continue;
      if (dump_loop_only && !node->inLoopBody() && !node->containsLoopBody())
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

} // namespace mlir::triton::gpu::partition_scheduling_detail
