#ifndef PROTON_DATA_METRIC_H_
#define PROTON_DATA_METRIC_H_

#include "Utility/Traits.h"
#include <variant>
#include <vector>

namespace proton {

enum class MetricKind { Flexible, Kernel, Count };

using MetricValueType = std::variant<uint64_t, int64_t, double>;

/// A metric is a class that can be associated with a context.
/// `Metric` is the base class for all metrics.
/// Each `Metric` has a name and a set of values.
/// Each value could be of type `uint64_t`, `int64_t`, or `double`,
/// Each value also has its own name and is either aggregatable or not.
/// Currently the only aggregation operation is `addition`.
class Metric {
public:
  Metric(MetricKind kind, size_t size) : kind(kind), values(size) {}

  virtual ~Metric() = default;

  virtual const std::string getName() const = 0;

  virtual const std::string getValueName(int valueId) const = 0;

  virtual bool isAggregatable(int valueId) const = 0;

  std::vector<MetricValueType> getValues() const { return values; }

  template <
      typename T,
      std::enable_if_t<is_one_of<T, uint64_t, int64_t, double>::value, int> = 0>
  T getValue(int valueId) {
    return std::get<T>(values[valueId]);
  }

  template <
      typename T,
      std::enable_if_t<is_one_of<T, uint64_t, int64_t, double>::value, int> = 0>
  void setValue(int valueId, T value) {
    values[valueId] = value;
  }

  template <
      typename T,
      std::enable_if_t<is_one_of<T, MetricValueType, Metric>::value, int> = 0>
  void updateValue(T &other) {
    for (int i = 0; i < values.size(); ++i) {
      if constexpr (std::is_same_v<T, Metric>) {
        // Assuming you want to use a member of `Metric` when T is `Metric`
        // and that `Metric` has a member `values` which is compatible with the
        // operation
        std::visit(
            [&](auto &&v, auto &&otherV) {
              if (isAggregatable(i)) {
                v += otherV;
              } else {
                v = otherV;
              }
            },
            values[i], other.values[i]);
      } else {
        // For non-Metric types, apply the value directly
        std::visit(
            [&](auto &&v, auto &&otherV) {
              if (isAggregatable(i)) {
                v += otherV;
              } else {
                v = otherV;
              }
            },
            values[i], other);
      }
    }
  }

  MetricKind getKind() const { return kind; }

private:
  const MetricKind kind;
  const std::string name;

protected:
  std::vector<MetricValueType> values;
};

/// A flexible metric is provided by users but not the hardware.
class FlexibleMetric : public Metric {
public:
  FlexibleMetric(const std::string &valueName,
                 std::variant<MetricValueType> value)
      : valueName(valueName), Metric(MetricKind::Flexible, 1) {
    std::visit([&](auto &&v) { this->values[0] = v; }, value);
  }

  const std::string getName() const override { return "FlexibleMetric"; }

  const std::string getValueName(int valueId) const override {
    return valueName;
  }

  bool isAggregatable(int valueId) const override { return AGGREGATABLE; }

private:
  // XXX(Keren): Currently all flexible metrics are aggregatable
  const static inline bool AGGREGATABLE = true;
  const std::string valueName;
};

class KernelMetric : public Metric {
public:
  enum kernelMetricKind : int {
    StartTime,
    EndTime,
    Invocations,
    Duration,
    Count,
  };

  KernelMetric() : Metric(MetricKind::Kernel, kernelMetricKind::Count) {}

  KernelMetric(uint64_t startTime, uint64_t endTime, uint64_t invocations)
      : KernelMetric() {
    this->values[StartTime] = startTime;
    this->values[EndTime] = endTime;
    this->values[Invocations] = invocations;
    this->values[Duration] = endTime - startTime;
  }

  virtual const std::string getName() const { return "KernelMetric"; }

  virtual const std::string getValueName(int valueId) const {
    return VALUE_NAMES[valueId];
  }

  virtual bool isAggregatable(int valueId) const {
    return AGGREGATABLE[valueId];
  }

private:
  const static inline bool AGGREGATABLE[kernelMetricKind::Count] = {
      false, false, true, true};
  const static inline std::string VALUE_NAMES[kernelMetricKind::Count] = {
      "StartTime (ns)", "EndTime (ns)", "Count", "Time (ns)"};
};

} // namespace proton

#endif // PROTON_DATA_METRIC_H_
