#ifndef PROTON_DATA_TRACE_DATA_H_
#define PROTON_DATA_TRACE_DATA_H_

#include "Data.h"
#include <iosfwd>
#include <memory>
#include <thread>
#include <unordered_map>

namespace proton {

class TraceData : public Data {
public:
  TraceData(const std::string &path, ContextSource *contextSource = nullptr);
  virtual ~TraceData();

  DataEntry addOp(size_t phase, size_t eventId,
                  const std::vector<Context> &contexts) override;

  void
  addMetrics(size_t scopeId,
             const std::map<std::string, MetricValueType> &metrics) override;

  class Trace;

protected:
  // ScopeInterface
  void enterScope(const Scope &scope) override final;

  void exitScope(const Scope &scope) override final;

private:
  template <typename CycleHandler, typename KernelHandler>
  void withTraceData(size_t phase, CycleHandler &&onCycleTrace,
                     KernelHandler &&onTraceData) const;

  // Data
  SerializedData doSerialize(OutputFormat outputFormat,
                             size_t phase) const override;

  OutputFormat getDefaultOutputFormat() const override {
    return OutputFormat::ChromeTrace;
  }

  std::string toJsonString(size_t phase) const;
  std::vector<uint8_t> toPerfettoTrace(size_t phase) const;
  void dumpChromeTrace(std::ostream &os, size_t phase) const;
  void dumpPerfettoTrace(std::ostream &os, size_t phase) const;
  size_t getCurrentThreadTraceId();

  PhaseStore<Trace> tracePhases;
  // ScopeId -> EventId
  std::unordered_map<size_t, size_t> scopeIdToEventId;
  // ThreadId -> TraceId
  std::unordered_map<std::thread::id, uint64_t> threadIdToTraceId;
  uint64_t nextThreadTraceId = 0;
};

} // namespace proton

#endif // PROTON_DATA_TRACE_DATA_H_
