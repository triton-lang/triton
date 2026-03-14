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
  /// Seals a step-buffer slot by recording a completion event on the compute
  /// stream and queuing the slot for async draining.
  void markStep(uint64_t streamId, uint64_t stepBufferToken) override;
  /// Makes the compute stream wait only when an in-flight step-buffer slot is
  /// about to be reused.
  void waitStepBuffer(uint64_t streamId, uint64_t stepBufferToken) override;

  // OpInterface
  void startOp(const Scope &scope) override {
    for (auto data : dataSet) {
      dataToEntryMap.insert_or_assign(data, data->addOp(scope.name));
    }
  }
  void stopOp(const Scope &scope) override { dataToEntryMap.clear(); }

private:
  /// One instrumented kernel launch waiting to be associated with a sealed
  /// step-buffer slot and drained later.
  struct PendingInstrumentedOp {
    uint64_t streamId;
    uint64_t functionId;
    uint8_t *buffer;
    size_t size;
    size_t stepId;
    uint64_t deviceId;
    DataToEntryMap dataToEntryMap;
  };

  /// A sealed step-buffer slot whose compute-stream completion event has been
  /// recorded but whose async copy has not been scheduled yet.
  struct PendingStepFence {
    size_t stepId;
    uint64_t stepBufferToken;
    void *copyStream;
    void *completionEvent;
  };

  /// A step-buffer slot with an in-flight async D2H copy into a pinned host
  /// staging buffer.
  struct InFlightInstrumentedStep {
    size_t stepId;
    uint64_t stepBufferToken;
    std::vector<PendingInstrumentedOp> pendingOps;
    uint8_t *hostBuffer;
    size_t hostBufferSize;
    void *copyStream;
    void *completionEvent;
  };

  std::shared_ptr<ParserConfig> getParserConfig(uint64_t functionId,
                                                size_t bufferSize) const;
  /// Computes how many bytes of a sealed step-buffer slot must be copied to
  /// cover all pending launches in that step.
  size_t getStepCopySize(const std::vector<PendingInstrumentedOp> &pendingOps,
                         uint64_t stepBufferToken) const;
  /// Reuses or allocates a pinned host staging buffer large enough for one
  /// copied step-buffer slot.
  uint8_t *acquireHostStagingBuffer(size_t size);
  /// Returns a pinned host staging buffer to the reusable pool keyed by size.
  void releaseHostStagingBuffer(uint8_t *buffer, size_t size);
  /// Turns sealed step-buffer slots into async copy work on the per-device
  /// copy stream.
  void scheduleReadySteps();
  /// Parses any completed async copies, optionally blocking until all in-flight
  /// copy work has finished.
  void processCompletedCopies(bool blockUntilComplete);
  /// Converts one copied launch record or scratch slice into profiler data.
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
  // Launches waiting for their enclosing step buffer to be sealed.
  std::vector<PendingInstrumentedOp> pendingInstrumentedOps;
  // Sealed step-buffer slots waiting to be scheduled on a copy stream.
  std::vector<PendingStepFence> pendingStepFences;
  // Step-buffer slots currently being copied into pinned host staging buffers.
  std::vector<InFlightInstrumentedStep> inflightInstrumentedSteps;
  // Reusable pinned host staging buffers, keyed by capacity.
  std::multimap<size_t, uint8_t *> availableHostStagingBuffers;
};

} // namespace proton

#endif // PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_
