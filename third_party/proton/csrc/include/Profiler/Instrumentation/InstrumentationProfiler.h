#ifndef PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_
#define PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_

#include "Context/Context.h"
#include "Device.h"
#include "Metadata.h"
#include "Profiler/Profiler.h"
#include "Runtime/Runtime.h"
#include "TraceDataIO/Parser.h"
#include "Utility/Singleton.h"
#include <vector>

namespace proton {

class InstrumentationProfiler : public Profiler,
                                public InstrumentationInterface,
                                public OpInterface,
                                public Singleton<InstrumentationProfiler> {
public:
  InstrumentationProfiler() = default;
  virtual ~InstrumentationProfiler();

  std::vector<uint64_t> drainCompletedBufferPtrs();

protected:
  // Profiler
  virtual void doStart() override;
  virtual void doFlush() override;
  virtual void doStop() override;
  virtual void
  doSetMode(const std::vector<std::string> &modeAndOptions) override;
  virtual void doAddMetrics(
      size_t scopeId,
      const std::map<std::string, MetricValueType> &scalarMetrics,
      const std::map<std::string, TensorMetric> &tensorMetrics) override;

  // InstrumentationInterface
  void initFunctionMetadata(
      uint64_t functionId, const std::string &functionName,
      const std::vector<std::pair<size_t, std::string>> &scopeIdNames,
      const std::vector<std::pair<size_t, size_t>> &scopeIdParentIds,
      const std::string &metadataPath) override;
  void destroyFunctionMetadata(uint64_t functionId) override;
  void enterInstrumentedOp(uint64_t streamId, uint64_t functionId,
                           uint8_t *buffer, size_t size) override;
  void exitInstrumentedOp(uint64_t streamId, uint64_t functionId,
                          uint8_t *buffer, size_t size) override;
  void markStep(uint64_t streamId, uint64_t stepBufferToken) override;
  void waitStepBuffer(uint64_t streamId, uint64_t stepBufferToken) override;

  // OpInterface
  void startOp(const Scope &scope) override {
    for (auto data : dataSet) {
      dataToEntryMap.insert_or_assign(data, data->addOp(scope.name));
    }
  }
  void stopOp(const Scope &scope) override { dataToEntryMap.clear(); }

private:
  struct PendingInstrumentedOp {
    uint64_t streamId;
    uint64_t functionId;
    uint8_t *buffer;
    size_t size;
    size_t stepId;
    uint64_t deviceId;
    DataToEntryMap dataToEntryMap;
  };

  struct PendingStepFence {
    size_t stepId;
    uint64_t stepBufferToken;
    void *copyStream;
    void *completionEvent;
  };

  struct InFlightInstrumentedStep {
    size_t stepId;
    uint64_t stepBufferToken;
    std::vector<PendingInstrumentedOp> pendingOps;
    std::vector<uint8_t *> hostBuffers;
    void *copyStream;
    void *completionEvent;
  };

  std::shared_ptr<ParserConfig> getParserConfig(uint64_t functionId,
                                                size_t bufferSize) const;
  void scheduleReadySteps();
  void processCompletedCopies(bool blockUntilComplete);
  void parseCopiedInstrumentedOp(const PendingInstrumentedOp &pendingOp,
                                 uint8_t *hostBuffer, size_t size);

  Runtime *runtime;
  // device -> flush/copy stream
  std::map<void *, void *> deviceStreams;
  std::map<std::string, std::string> modeOptions;
  // functionId -> scopeId -> scopeName
  std::map<uint64_t, std::map<size_t, std::string>> functionScopeIdNames;
  // functionId -> scopeId -> contexts
  std::map<uint64_t, std::map<size_t, std::vector<Context>>>
      functionScopeIdContexts;
  ;
  // functionId -> functionName
  std::map<uint64_t, std::string> functionNames;
  // functionId -> metadata
  std::map<uint64_t, InstrumentationMetadata> functionMetadata;
  // Active per-data entries for the current op.
  DataToEntryMap dataToEntryMap;
  size_t currentStepId{0};
  std::vector<PendingInstrumentedOp> pendingInstrumentedOps;
  std::vector<PendingStepFence> pendingStepFences;
  std::vector<InFlightInstrumentedStep> inflightInstrumentedSteps;
  std::vector<uint64_t> completedBufferPtrs;
};

} // namespace proton

#endif // PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_
