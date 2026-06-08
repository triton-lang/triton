#include "Data/TreeData.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Device.h"
#include "DeviceType.h"
#include "Dump/TreeDataDump.h"
#include "Profiler/Graph.h"
#include "Utility/Errors.h"
#include "Utility/MsgPackWriter.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace proton {

class TreeData::Tree {
public:
  struct TreeNode : public Context {
    inline static const size_t RootId = 0;
    inline static const size_t DummyId = std::numeric_limits<size_t>::max();

    struct ChildEntry {
      std::string_view name;
      size_t id = DummyId;
    };

    TreeNode() = default;
    explicit TreeNode(size_t id, const std::string &name)
        : id(id), Context(name) {}
    TreeNode(size_t id, size_t parentId, const std::string &name)
        : id(id), parentId(parentId), Context(name) {}
    TreeNode(size_t id, size_t parentId, const Context &context)
        : Context(context), parentId(parentId), id(id) {}
    virtual ~TreeNode() = default;

    void addChild(std::string_view childName, size_t id) {
      children.push_back({childName, id});
      childIndex.emplace(childName, id);
    }

    size_t findChild(std::string_view childName) const {
      auto it = childIndex.find(childName);
      return it != childIndex.end() ? it->second : DummyId;
    }

    size_t parentId = DummyId;
    size_t id = DummyId;
    std::vector<ChildEntry> children = {};
    std::unordered_map<std::string_view, size_t> childIndex = {};
    // Direct and linked metrics associated with this tree node.
    DataEntry::MetricSet metricSet{};
    friend class Tree;
  };

  Tree() { treeNodes.emplace_back(TreeNode::RootId, TreeNode::RootId, "ROOT"); }

  size_t addNode(const std::vector<Context> &contexts, size_t parentId) {
    for (const auto &context : contexts) {
      parentId = addNode(context, parentId);
    }
    return parentId;
  }

  size_t addNode(const Context &context, size_t parentId) {
    auto &parent = getNode(parentId);
    std::string_view contextName = context.name;
    auto existingChildId = parent.findChild(contextName);
    if (existingChildId != TreeNode::DummyId)
      return existingChildId;
    auto id = nextContextId++;
    treeNodes.emplace_back(id, parentId, context);
    parent.addChild(treeNodes.back().name, id);
    return id;
  }

  size_t addNode(const std::vector<Context> &indices) {
    auto parentId = TreeNode::RootId;
    for (auto index : indices) {
      parentId = addNode(index, parentId);
    }
    return parentId;
  }

  TreeNode &getNode(size_t id) { return treeNodes.at(id); }

  void upsertFlexibleMetric(size_t contextId,
                            const FlexibleMetric &flexibleMetric) {
    auto &node = getNode(contextId);
    auto &flexibleMetrics = node.metricSet.flexibleMetrics;
    auto valueName = std::string(flexibleMetric.getValueName(0));
    auto it = flexibleMetrics.find(valueName);
    if (it == flexibleMetrics.end()) {
      flexibleMetrics.emplace(std::move(valueName), flexibleMetric);
    } else {
      it->second.updateMetric(flexibleMetric);
    }
  }

  enum class WalkPolicy { PreOrder, PostOrder };

  template <WalkPolicy walkPolicy, typename FnT> void walk(FnT &&fn) {
    if constexpr (walkPolicy == WalkPolicy::PreOrder) {
      walkPreOrder(TreeNode::RootId, fn);
    } else if constexpr (walkPolicy == WalkPolicy::PostOrder) {
      walkPostOrder(TreeNode::RootId, fn);
    }
  }

  template <typename FnT> void walkPreOrder(size_t contextId, FnT &&fn) {
    fn(getNode(contextId));
    for (const auto &child : getNode(contextId).children) {
      walkPreOrder(child.id, fn);
    }
  }

  size_t size() const { return treeNodes.size(); }

private:
  size_t nextContextId = TreeNode::RootId + 1;
  // Node ids are dense and assigned sequentially, so index lookup is enough.
  std::deque<TreeNode> treeNodes;
};

} // namespace proton

#include "Dump/Hatchet.cpp"
#include "Dump/HatchetMsgPack.cpp"
#include "Dump/TreeDataDump.cpp"

namespace proton {

void TreeData::enterScope(const Scope &scope) {
  // enterOp and addMetric maybe called from different threads
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto *currentTree = currentPhasePtrAs<Tree>();
  std::vector<Context> contexts;
  if (contextSource != nullptr)
    contexts = contextSource->getContexts();
  else
    contexts.emplace_back(scope.name);
  auto contextId = currentTree->addNode(contexts);
  scopeIdToContextId[scope.scopeId] = contextId;
}

void TreeData::exitScope(const Scope &scope) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  scopeIdToContextId.erase(scope.scopeId);
}

DataEntry TreeData::addOp(size_t phase, size_t contextId,
                          const std::vector<Context> &contexts) {
  auto lock = lockIfCurrentOrVirtualPhase(phase);
  if (contextId == Data::kRootEntryId) {
    contextId = Tree::TreeNode::RootId;
  }
  auto *tree = phasePtrAs<Tree>(phase);
  auto newContextId = tree->addNode(contexts, contextId);
  auto &node = tree->getNode(newContextId);
  return DataEntry(newContextId, phase, node.metricSet);
}

void TreeData::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &metrics) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto *currentTree = currentPhasePtrAs<Tree>();
  auto contextId = scopeIdToContextId.at(scopeId);
  for (auto [metricName, metricValue] : metrics) {
    currentTree->upsertFlexibleMetric(contextId,
                                      FlexibleMetric(metricName, metricValue));
  }
}

void TreeData::doDump(std::ostream &os, OutputFormat outputFormat,
                      size_t phase) const {
  if (outputFormat == OutputFormat::Hatchet) {
    dumpHatchet(os, phase);
  } else if (outputFormat == OutputFormat::HatchetMsgPack) {
    dumpHatchetMsgPack(os, phase);
  } else {
    throw makeInvalidArgument("Output format not supported");
  }
}

TreeData::TreeData(const std::string &path, ContextSource *contextSource)
    : Data(path, contextSource) {
  initPhaseStore(treePhases);
}

TreeData::~TreeData() {}

} // namespace proton
