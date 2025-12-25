#ifndef PROTON_DATA_TRACE_DATA_H_
#define PROTON_DATA_TRACE_DATA_H_

#include "Data.h"
#include <memory>
#include <unordered_map>

namespace proton {

class TraceData : public Data {
public:
  TraceData(const std::string &path, ContextSource *contextSource = nullptr);
  virtual ~TraceData();

  DataEntry addOp(const std::string &name) override;

  DataEntry addOp(size_t eventId,
                  const std::vector<Context> &contexts) override;

  void addScopeMetrics(
      size_t scopeId,
      const std::map<std::string, MetricValueType> &metrics) override;

  void addEntryMetrics(
      size_t entryId,
      const std::map<std::string, MetricValueType> &metrics) override;

  void clear() override;

  std::string toJsonString() const override;

  std::vector<uint8_t> toMsgPack() const override;

  class Trace;

protected:
  // ScopeInterface
  void enterScope(const Scope &scope) override final;

  void exitScope(const Scope &scope) override final;

private:
  void doDump(std::ostream &os, OutputFormat outputFormat) const override;
  void dumpChromeTrace(std::ostream &os) const;

  OutputFormat getDefaultOutputFormat() const override {
    return OutputFormat::ChromeTrace;
  }

  std::unique_ptr<Trace> trace;
  // ScopeId -> EventId
  std::unordered_map<size_t, size_t> scopeIdToEventId;
};

} // namespace proton

#endif // PROTON_DATA_TRACE_DATA_H_
