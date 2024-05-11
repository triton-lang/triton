#ifndef PROTON_DATA_TREE_DATA_H_
#define PROTON_DATA_TREE_DATA_H_

#include "Context/Context.h"
#include "Data.h"
#include <stdexcept>

namespace proton {

class TreeData : public Data {
public:
  TreeData(const std::string &path, ContextSource *contextSource)
      : Data(path, contextSource) {
    init();
  }

  TreeData(const std::string &path) : TreeData(path, nullptr) {}

  void addMetric(size_t scopeId, std::shared_ptr<Metric> metric) override;

  void addMetrics(size_t scopeId,
                  const std::map<std::string, MetricValueType> &metrics,
                  bool aggregable) override;

protected:
  // OpInterface
  void startOp(const Scope &scope) override;

  void stopOp(const Scope &scope) override;

private:
  class Tree {
  public:
    struct TreeNode : public Context {
      inline static const size_t RootId = 0;
      inline static const size_t DummyId = std::numeric_limits<size_t>::max();

      TreeNode() = default;
      explicit TreeNode(size_t id, const std::string &name)
          : id(id), Context(name) {}
      TreeNode(size_t id, size_t parentId, const std::string &name)
          : id(id), parentId(parentId), Context(name) {}
      virtual ~TreeNode() = default;

      void addChild(const Context &context, size_t id) {
        children[context] = id;
      }

      bool hasChild(const Context &context) const {
        return children.find(context) != children.end();
      }

      size_t getChild(const Context &context) const {
        return children.at(context);
      }

      size_t parentId = DummyId;
      size_t id = DummyId;
      std::map<Context, size_t> children = {};
      std::map<MetricKind, std::shared_ptr<Metric>> metrics = {};
      std::map<std::string, FlexibleMetric> flexibleMetrics = {};
      friend class Tree;
    };

    Tree() {
      treeNodeMap.try_emplace(TreeNode::RootId, TreeNode::RootId, "ROOT");
    }

    size_t addNode(const Context &context, size_t parentId) {
      if (treeNodeMap[parentId].hasChild(context)) {
        return treeNodeMap[parentId].getChild(context);
      }
      auto id = nextContextId++;
      treeNodeMap.try_emplace(id, id, parentId, context.name);
      treeNodeMap[parentId].addChild(context, id);
      return id;
    }

    size_t addNode(const std::vector<Context> &indices) {
      if (indices.empty()) {
        throw std::runtime_error("Indices is empty");
      }
      auto parentId = TreeNode::RootId;
      for (auto index : indices) {
        parentId = addNode(index, parentId);
      }
      return parentId;
    }

    TreeNode &getNode(size_t id) { return treeNodeMap.at(id); }

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
      for (auto &child : getNode(contextId).children) {
        walkPreOrder(child.second, fn);
      }
    }

    template <typename FnT> void walkPostOrder(size_t contextId, FnT &&fn) {
      for (auto &child : getNode(contextId).children) {
        walkPostOrder(child.second, fn);
      }
      fn(getNode(contextId));
    }

  private:
    size_t nextContextId = TreeNode::RootId + 1;
    // tree node id->tree node
    std::map<size_t, TreeNode> treeNodeMap;
  };

  void init();
  void dumpHatchet(std::ostream &os) const;
  void doDump(std::ostream &os, OutputFormat outputFormat) const override;

  std::unique_ptr<Tree> tree;
  // ScopeId -> ContextId
  std::map<size_t, size_t> scopeIdToContextId;
};

} // namespace proton

#endif // PROTON_DATA_TREE_DATA_H_
