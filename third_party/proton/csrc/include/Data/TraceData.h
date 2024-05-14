#ifndef PROTON_DATA_TRACE_DATA_H_
#define PROTON_DATA_TRACE_DATA_H_

#include "Data.h"

namespace proton {

class TraceData : public Data {
public:
  using Data::Data;

  void addMetric(size_t scopeId, std::shared_ptr<Metric> metric) override;

  void addMetrics(size_t scopeId,
                  const std::map<std::string, MetricValueType> &metrics,
                  bool aggregable) override;

protected:
  void startOp(const Scope &scope) override final;

  void stopOp(const Scope &scope) override final;

private:
  void doDump(std::ostream &os, OutputFormat outputFormat) const override;
};

} // namespace proton

#endif // PROTON_DATA_TRACE_DATA_H_
