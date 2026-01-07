#ifndef PROTON_DATA_TREE_DATA_H_
#define PROTON_DATA_TREE_DATA_H_

#include "Context/Context.h"
#include "Data.h"
#include "nlohmann/json.hpp"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

namespace proton {

class TreeData : public Data {
public:
  TreeData(const std::string &path, ContextSource *contextSource);
  virtual ~TreeData();

  TreeData(const std::string &path) : TreeData(path, nullptr) {}

  DataEntry addOp(const std::string &name) override;

  DataEntry addOp(size_t contextId,
                  const std::vector<Context> &contexts) override;

  void addScopeMetrics(
      size_t scopeId,
      const std::map<std::string, MetricValueType> &metrics) override;

  void addEntryMetrics(
      size_t entryId,
      const std::map<std::string, MetricValueType> &metrics) override;

  std::vector<uint8_t> toMsgPack() const override;

  std::string toJsonString() const override;

  void clear() override;

protected:
  // ScopeInterface
  void enterScope(const Scope &scope) override;

  void exitScope(const Scope &scope) override;

private:
  // `tree` and `scopeIdToContextId` can be accessed by both the user thread and
  // the background threads concurrently, so methods that access them should be
  // protected by a (shared) mutex.
  class Tree;
  json buildHatchetJson(TreeData::Tree *tree) const;
  std::vector<uint8_t> buildHatchetMsgPack(TreeData::Tree *tree) const;

  void doDump(std::ostream &os, OutputFormat outputFormat) const override;
  OutputFormat getDefaultOutputFormat() const override {
    return OutputFormat::Hatchet;
  }

  void dumpHatchet(std::ostream &os) const;

  std::unique_ptr<Tree> tree;
  // ScopeId -> ContextId
  std::unordered_map<size_t, size_t> scopeIdToContextId;
};

} // namespace proton

#endif // PROTON_DATA_TREE_DATA_H_
