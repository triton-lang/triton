#include "Data/Data.h"
#include "Utility/String.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include <shared_mutex>

namespace proton {

void DataEntry::upsertMetric(std::unique_ptr<Metric> metric) const {
  auto &metrics = metricSet.get().metrics;
  auto it = metrics.find(metric->getKind());
  if (it == metrics.end()) {
    metrics.emplace(metric->getKind(), std::move(metric));
  } else {
    it->second->updateMetric(*metric);
  }
}

void DataEntry::upsertLinkedMetric(std::unique_ptr<Metric> metric,
                                   size_t linkedId) const {
  auto &linkedMetrics = metricSet.get().linkedMetrics;
  auto &linkedMetricMap = linkedMetrics[linkedId];
  auto it = linkedMetricMap.find(metric->getKind());
  if (it == linkedMetricMap.end()) {
    linkedMetricMap.emplace(metric->getKind(), std::move(metric));
  } else {
    it->second->updateMetric(*metric);
  }
}

void DataEntry::upsertFlexibleMetric(const std::string &metricName,
                                     const MetricValueType &metricValue) const {
  auto &flexibleMetrics = metricSet.get().flexibleMetrics;
  auto it = flexibleMetrics.find(metricName);
  if (it == flexibleMetrics.end()) {
    flexibleMetrics.emplace(metricName,
                            FlexibleMetric(metricName, metricValue));
  } else {
    it->second.updateValue(metricValue);
  }
}

void DataEntry::upsertFlexibleMetrics(
    const std::map<std::string, MetricValueType> &metrics) const {
  for (const auto &[metricName, metricValue] : metrics) {
    upsertFlexibleMetric(metricName, metricValue);
  }
}

void Data::initPhaseStore(PhaseStoreBase &store) {
  phaseStore = &store;
  currentPhasePtr = phaseStore->createPtr(0);
  phaseStore->createPtr(kVirtualPhase);
  activePhases.insert(0);
}

DataEntry Data::addOp(const std::string &opName) {
  std::vector<Context> contexts;
  if (contextSource != nullptr)
    contexts = contextSource->getContexts();
  if (!opName.empty())
    contexts.emplace_back(opName);
  const auto phase = currentPhase.load(std::memory_order_relaxed);
  return addOp(phase, kRootEntryId, contexts);
}

size_t Data::advancePhase() {
  std::unique_lock<std::shared_mutex> lock(mutex);
  const auto nextPhase = currentPhase.load(std::memory_order_relaxed) + 1;
  currentPhasePtr = phaseStore->createPtr(nextPhase);
  activePhases.insert(nextPhase);
  currentPhase.store(nextPhase, std::memory_order_release);
  return nextPhase;
}

void Data::clear(size_t phase, bool clearUpToPhase) {
  // No locking needed.
  // If phase == currentPhase, we expect users to call clear right after
  // deactivating the profiler, without any GPU events in between.
  // If phase < currentPhase, clearing a past phase is safe without locks.
  if (clearUpToPhase)
    phaseStore->clearUpToInclusive(phase);
  else
    phaseStore->clearPhase(phase);

  std::unique_lock<std::shared_mutex> lock(mutex);
  if (clearUpToPhase) {
    for (auto it = activePhases.begin(); it != activePhases.end();) {
      if (*it <= phase) {
        it = activePhases.erase(it);
      } else {
        ++it;
      }
    }
  } else {
    activePhases.erase(phase);
  }

  // In case the current phase is cleared, recreate its pointer.
  const auto phaseToRecreate = currentPhase.load(std::memory_order_relaxed);
  currentPhasePtr = phaseStore->createPtr(phaseToRecreate);
  activePhases.insert(phaseToRecreate);
}

void Data::completePhase(size_t phase) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  if (completeUpToPhase == kNoCompletePhase || phase > completeUpToPhase)
    completeUpToPhase = phase;
}

Data::PhaseInfo Data::getPhaseInfo() const {
  std::shared_lock<std::shared_mutex> lock(mutex);
  return PhaseInfo{currentPhase.load(std::memory_order_relaxed),
                   completeUpToPhase};
}

void Data::dump(const std::string &outputFormat) {
  std::shared_lock<std::shared_mutex> lock(mutex);

  OutputFormat outputFormatEnum = outputFormat.empty()
                                      ? getDefaultOutputFormat()
                                      : parseOutputFormat(outputFormat);

  for (auto phase : activePhases) {
    std::unique_ptr<std::ostream> out;
    if (path.empty() || path == "-") {
      out.reset(new std::ostream(std::cout.rdbuf())); // Redirecting to cout
    } else {
      auto suffix = currentPhase.load(std::memory_order_relaxed) == 0
                        ? ""
                        : ".part_" + std::to_string(phase);
      const auto filePath =
          path + suffix + "." + outputFormatToString(outputFormatEnum);
      const auto fileMode =
          (outputFormatEnum == OutputFormat::HatchetMsgPack)
              ? (std::ios::out | std::ios::binary | std::ios::trunc)
              : (std::ios::out | std::ios::trunc);
      out.reset(
          new std::ofstream(filePath, fileMode)); // Opening a file for output
    }
    doDump(*out, outputFormatEnum, phase);
  }
}

OutputFormat parseOutputFormat(const std::string &outputFormat) {
  if (toLower(outputFormat) == "hatchet") {
    return OutputFormat::Hatchet;
  } else if (toLower(outputFormat) == "hatchet_msgpack") {
    return OutputFormat::HatchetMsgPack;
  } else if (toLower(outputFormat) == "chrome_trace") {
    return OutputFormat::ChromeTrace;
  } else {
    throw std::runtime_error("Unknown output format: " + outputFormat);
  }
}

const std::string outputFormatToString(OutputFormat outputFormat) {
  if (outputFormat == OutputFormat::Hatchet) {
    return "hatchet";
  } else if (outputFormat == OutputFormat::HatchetMsgPack) {
    return "hatchet_msgpack";
  } else if (outputFormat == OutputFormat::ChromeTrace) {
    return "chrome_trace";
  }
  throw std::runtime_error("Unknown output format: " +
                           std::to_string(static_cast<int>(outputFormat)));
}

} // namespace proton
