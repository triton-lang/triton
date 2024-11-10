#ifndef PROTON_DATA_TREE_DATA_H_
#define PROTON_DATA_TREE_DATA_H_

#include "Context/Context.h"
#include "Data.h"
#include <stdexcept>
#include <unordered_map>

namespace proton {

class TreeData : public Data {
public:
  TreeData(const std::string &path, ContextSource *contextSource);
  virtual ~TreeData();

  TreeData(const std::string &path) : TreeData(path, nullptr) {}

  size_t addScope(size_t scopeId, const std::string &name) override final;

  void addMetric(size_t scopeId, std::shared_ptr<Metric> metric) override final;

  void addMetrics(size_t scopeId,
                  const std::map<std::string, MetricValueType> &metrics,
                  bool aggregable) override final;

protected:
  // OpInterface
  void startOp(const Scope &scope) override final;

  void stopOp(const Scope &scope) override final;

private:
  void init();
  void dumpHatchet(std::ostream &os) const;
  void doDump(std::ostream &os, OutputFormat outputFormat) const override final;

  class Tree;
  std::unique_ptr<Tree> tree;
  // ScopeId -> ContextId
  std::unordered_map<size_t, size_t> scopeIdToContextId;
};

} // namespace proton

#endif // PROTON_DATA_TREE_DATA_H_
