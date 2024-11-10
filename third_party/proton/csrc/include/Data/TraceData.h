#ifndef PROTON_DATA_TRACE_DATA_H_
#define PROTON_DATA_TRACE_DATA_H_

#include "Data.h"

namespace proton {

class TraceData : public Data {
public:
  using Data::Data;
  virtual ~TraceData() = default;

  size_t addScope(size_t scopeId, const std::string &name) override final;

  void addMetric(size_t scopeId, std::shared_ptr<Metric> metric) override final;

  void addMetrics(size_t scopeId,
                  const std::map<std::string, MetricValueType> &metrics,
                  bool aggregable) override final;

  void flush() override final;

protected:
  void startOp(const Scope &scope) override final;

  void stopOp(const Scope &scope) override final;

private:
  void doDump(std::ostream &os, OutputFormat outputFormat) const override;
};

} // namespace proton

#endif // PROTON_DATA_TRACE_DATA_H_
