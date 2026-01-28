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

  std::string toJsonString(size_t phase) const override;

  std::vector<uint8_t> toMsgPack(size_t phase) const override;

  DataEntry addOp(const std::string &name) override;

  DataEntry addOp(size_t phase, size_t eventId,
                  const std::vector<Context> &contexts) override;

  void
  addMetrics(size_t scopeId,
             const std::map<std::string, MetricValueType> &metrics) override;

  void
  addMetrics(size_t phase, size_t entryId,
             const std::map<std::string, MetricValueType> &metrics) override;

  class Trace;

protected:
  // ScopeInterface
  void enterScope(const Scope &scope) override final;

  void exitScope(const Scope &scope) override final;

private:
  // Data
  void doDump(std::ostream &os, OutputFormat outputFormat,
              size_t phase) const override;

  OutputFormat getDefaultOutputFormat() const override {
    return OutputFormat::ChromeTrace;
  }

  void dumpChromeTrace(std::ostream &os, size_t phase) const;

  PhaseStore<Trace> tracePhases;
  // ScopeId -> EventId
  std::unordered_map<size_t, size_t> scopeIdToEventId;
};

} // namespace proton

#endif // PROTON_DATA_TRACE_DATA_H_
