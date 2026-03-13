#include "Profiler/Instrumentation/InstrumentationProfiler.h"
#include "Data/TraceData.h"
#include "TraceDataIO/CircularLayoutParser.h"

#include "Runtime/CudaRuntime.h"
#include "Runtime/HipRuntime.h"
#include "Utility/Numeric.h"
#include "Utility/String.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace proton {

constexpr size_t MAX_HOST_BUFFER_SIZE = 4LL * 1024LL * 1024LL * 1024LL; // 4GB

InstrumentationProfiler::~InstrumentationProfiler() {}

std::vector<uint64_t> InstrumentationProfiler::drainCompletedBufferPtrs() {
  auto completedPtrs = std::move(completedBufferPtrs);
  completedBufferPtrs.clear();
  return completedPtrs;
}

void InstrumentationProfiler::doStart() {
  // Start the instrumentation profiler.
}

void InstrumentationProfiler::doFlush() {
  scheduleReadySteps();
  processCompletedCopies(/*blockUntilComplete=*/false);
}

void InstrumentationProfiler::doStop() {
  // Stop the instrumentation profiler.
  // Process any ready async copies without blocking the host first.
  scheduleReadySteps();
  processCompletedCopies(/*blockUntilComplete=*/false);

  // Finalization may happen without an explicit step fence for the last batch
  // of launches. Drain those remaining launches synchronously on stop so the
  // output is complete, even though steady-state step flushing stays async.
  for (auto &pendingOp : pendingInstrumentedOps) {
    if (pendingOp.size > MAX_HOST_BUFFER_SIZE) {
      throw std::runtime_error(
          "Buffer size " + std::to_string(pendingOp.size) +
          " exceeds the limit " + std::to_string(MAX_HOST_BUFFER_SIZE) +
          ", not supported yet in proton");
    }
    uint8_t *hostBuffer = nullptr;
    runtime->allocateHostBuffer(&hostBuffer, pendingOp.size);
    runtime->copyDeviceToHostAsync(hostBuffer, pendingOp.buffer, pendingOp.size,
                                   reinterpret_cast<void *>(pendingOp.streamId));
    runtime->synchronizeStream(reinterpret_cast<void *>(pendingOp.streamId));
    parseCopiedInstrumentedOp(pendingOp, hostBuffer, pendingOp.size);
    runtime->freeHostBuffer(hostBuffer);
  }
  pendingInstrumentedOps.clear();

  // Finish any scheduled copies that are still in flight and parse them.
  processCompletedCopies(/*blockUntilComplete=*/true);

  for (auto &stepFence : pendingStepFences) {
    runtime->destroyEvent(stepFence.completionEvent);
  }
  pendingStepFences.clear();
  for (auto &[device, deviceStream] : deviceStreams) {
    runtime->destroyStream(deviceStream);
  }
  deviceStreams.clear();
  for (auto &[size, hostBuffer] : availableHostStagingBuffers) {
    (void)size;
    runtime->freeHostBuffer(hostBuffer);
  }
  availableHostStagingBuffers.clear();
  currentStepId = 0;
  completedBufferPtrs.clear();
  // Reset mode options
  modeOptions.clear();
  // Note that we don't clear function metadata and names here, as they may be
  // reused when the profiler is started again.
}

void InstrumentationProfiler::doSetMode(
    const std::vector<std::string> &modeAndOptions) {
  if (modeAndOptions.empty()) {
    throw std::runtime_error("Mode cannot be empty");
  }
  if (proton::toLower(modeAndOptions[0]) ==
      proton::toLower(DeviceTraits<DeviceType::CUDA>::name)) {
    runtime = &CudaRuntime::instance();
  } else if (proton::toLower(modeAndOptions[0]) ==
             proton::toLower(DeviceTraits<DeviceType::HIP>::name)) {
    runtime = &HipRuntime::instance();
  } else {
    throw std::runtime_error("Unknown device type: " + modeAndOptions[0]);
  }
  for (size_t i = 1; i < modeAndOptions.size(); ++i) {
    auto delimiterPos = modeAndOptions[i].find('=');
    if (delimiterPos != std::string::npos) {
      std::string key = modeAndOptions[i].substr(0, delimiterPos);
      std::string value = modeAndOptions[i].substr(delimiterPos + 1);
      modeOptions[key] = value;
    } else {
      modeOptions[modeAndOptions[i]] = "";
    }
  }
}
namespace {

std::vector<uint32_t>
getUnitIdVector(const std::map<std::string, std::string> &modeOptions,
                size_t totalUnits) {
  std::vector<uint32_t> unitIdVector;
  if (modeOptions.count("sampling_options") != 0) {
    auto &samplingOption = modeOptions.at("sampling_options");
    auto unitIds = proton::split(samplingOption, ",");
    for (auto uintId : unitIds) {
      if (proton::trim(uintId).empty()) {
        continue;
      }
      uint32_t id = std::stoi(uintId);
      unitIdVector.push_back(id);
    }
  }
  if (unitIdVector.empty()) {
    unitIdVector.resize(totalUnits);
    std::iota(unitIdVector.begin(), unitIdVector.end(), 0);
  }
  return unitIdVector;
}

bool isKernelTraceMode(const std::map<std::string, std::string> &modeOptions) {
  auto traceMode = modeOptions.find("trace_mode");
  return traceMode != modeOptions.end() && traceMode->second == "kernel";
}

} // namespace

std::shared_ptr<ParserConfig>
InstrumentationProfiler::getParserConfig(uint64_t functionId,
                                         size_t bufferSize) const {
  // Only support circular layout parser for now, but we will extend the support
  // to other parsers in the future
  auto config = std::make_shared<CircularLayoutParserConfig>();
  config->scratchMemSize =
      functionMetadata.at(functionId).getScratchMemorySize();
  if (!(modeOptions.count("granularity") == 0 ||
        modeOptions.at("granularity") == "GRANULARITY.WARP")) {
    throw std::runtime_error("Only warp granularity is supported for now");
  }
  config->totalUnits = functionMetadata.at(functionId).getNumWarps();
  config->numBlocks = bufferSize / config->scratchMemSize;
  config->uidVec = getUnitIdVector(modeOptions, config->totalUnits);

  // Check if the uidVec is valid
  for (auto uid : config->uidVec)
    if (uid >= config->totalUnits) {
      throw std::runtime_error(
          "Invalid sampling warp id: " + std::to_string(uid) + ". We have " +
          std::to_string(config->totalUnits) +
          " warps in total. Please check the proton sampling options.");
    }

  config->device = Device();
  config->device.type = runtime->getDeviceType();

  return config;
}

void InstrumentationProfiler::initFunctionMetadata(
    uint64_t functionId, const std::string &functionName,
    const std::vector<std::pair<size_t, std::string>> &scopeIdPairs,
    const std::vector<std::pair<size_t, size_t>> &scopeIdParentPairs,
    const std::string &metadataPath) {
  if (functionScopeIdNames.count(functionId)) {
    throw std::runtime_error(
        "Duplicate function id: " + std::to_string(functionId) +
        " for function " + functionName);
  }
  functionNames[functionId] = functionName;
  for (auto &pair : scopeIdPairs) {
    auto scopeId = pair.first;
    auto scopeName = pair.second;
    if (functionScopeIdNames[functionId].count(scopeId)) {
      throw std::runtime_error(
          "Duplicate scope id: " + std::to_string(scopeId) + " for function " +
          functionName);
    }
    functionScopeIdNames[functionId][scopeId] = scopeName;
  }
  // Synthesize the calling contexts
  std::map<size_t, size_t> scopeIdParentMap;
  for (auto &pair : scopeIdParentPairs) {
    auto scopeId = pair.first;
    auto parentId = pair.second;
    scopeIdParentMap[scopeId] = parentId;
  }
  for (auto &[scopeId, name] : functionScopeIdNames[functionId]) {
    std::vector<Context> contexts = {name};
    auto currentId = scopeId;
    while (scopeIdParentMap.count(currentId) > 0) {
      auto parentId = scopeIdParentMap[currentId];
      auto parentName = functionScopeIdNames[functionId].at(parentId);
      contexts.emplace_back(parentName);
      currentId = parentId;
    }
    std::reverse(contexts.begin(), contexts.end());
    functionScopeIdContexts[functionId][scopeId] = contexts;
  }
  functionMetadata.emplace(functionId, InstrumentationMetadata(metadataPath));
}

void InstrumentationProfiler::destroyFunctionMetadata(uint64_t functionId) {
  functionScopeIdNames.erase(functionId);
  functionScopeIdContexts.erase(functionId);
  functionNames.erase(functionId);
  functionMetadata.erase(functionId);
}

void InstrumentationProfiler::enterInstrumentedOp(uint64_t streamId,
                                                  uint64_t functionId,
                                                  uint8_t *buffer,
                                                  size_t size) {
  (void)streamId;
  (void)functionId;
  (void)buffer;
  (void)size;
}

void InstrumentationProfiler::scheduleReadySteps() {
  if (pendingStepFences.empty() || pendingInstrumentedOps.empty()) {
    return;
  }

  auto pendingOps = std::move(pendingInstrumentedOps);
  pendingInstrumentedOps.clear();

  std::map<size_t, std::vector<PendingInstrumentedOp>> opsByStep;
  for (auto &pendingOp : pendingOps) {
    opsByStep[pendingOp.stepId].push_back(std::move(pendingOp));
  }

  for (auto &stepFence : pendingStepFences) {
    auto readyOpsIt = opsByStep.find(stepFence.stepId);
    if (readyOpsIt == opsByStep.end()) {
      runtime->destroyEvent(stepFence.completionEvent);
      continue;
    }

    runtime->waitEvent(stepFence.copyStream, stepFence.completionEvent);
    std::vector<PendingInstrumentedOp> stepOps;
    stepOps.reserve(readyOpsIt->second.size());
    for (auto &pendingOp : readyOpsIt->second) {
      if (pendingOp.size > MAX_HOST_BUFFER_SIZE) {
        throw std::runtime_error(
            "Buffer size " + std::to_string(pendingOp.size) +
            " exceeds the limit " + std::to_string(MAX_HOST_BUFFER_SIZE) +
            ", not supported yet in proton");
      }
      stepOps.push_back(std::move(pendingOp));
    }
    auto stepCopySize = getStepCopySize(stepOps, stepFence.stepBufferToken);
    auto *hostBuffer = acquireHostStagingBuffer(stepCopySize);
    runtime->copyDeviceToHostAsync(
        hostBuffer, reinterpret_cast<uint8_t *>(stepFence.stepBufferToken),
        stepCopySize, stepFence.copyStream);
    auto *copyDoneEvent = runtime->createEvent();
    runtime->recordEvent(copyDoneEvent, stepFence.copyStream);
    inflightInstrumentedSteps.push_back(
        InFlightInstrumentedStep{stepFence.stepId, stepFence.stepBufferToken,
                                 std::move(stepOps), hostBuffer, stepCopySize,
                                 stepFence.copyStream, copyDoneEvent});
    opsByStep.erase(readyOpsIt);
    runtime->destroyEvent(stepFence.completionEvent);
  }

  pendingStepFences.clear();
  for (auto &[_, remainingOps] : opsByStep) {
    for (auto &pendingOp : remainingOps) {
      pendingInstrumentedOps.push_back(std::move(pendingOp));
    }
  }
}

size_t InstrumentationProfiler::getStepCopySize(
    const std::vector<PendingInstrumentedOp> &pendingOps,
    uint64_t stepBufferToken) const {
  size_t stepCopySize = 0;
  for (const auto &pendingOp : pendingOps) {
    auto bufferPtr = reinterpret_cast<uint64_t>(pendingOp.buffer);
    if (bufferPtr < stepBufferToken) {
      throw std::runtime_error(
          "Instrumented launch buffer does not belong to the step buffer");
    }
    auto opEnd = static_cast<size_t>(bufferPtr - stepBufferToken) + pendingOp.size;
    stepCopySize = std::max(stepCopySize, opEnd);
  }
  return stepCopySize;
}

uint8_t *InstrumentationProfiler::acquireHostStagingBuffer(size_t size) {
  auto it = availableHostStagingBuffers.lower_bound(size);
  if (it != availableHostStagingBuffers.end()) {
    auto *hostBuffer = it->second;
    availableHostStagingBuffers.erase(it);
    return hostBuffer;
  }

  uint8_t *hostBuffer = nullptr;
  runtime->allocateHostBuffer(&hostBuffer, size);
  return hostBuffer;
}

void InstrumentationProfiler::releaseHostStagingBuffer(uint8_t *buffer,
                                                       size_t size) {
  availableHostStagingBuffers.emplace(size, buffer);
}

void InstrumentationProfiler::processCompletedCopies(bool blockUntilComplete) {
  std::vector<InFlightInstrumentedStep> pendingInflightSteps;
  pendingInflightSteps.reserve(inflightInstrumentedSteps.size());
  for (auto &inflightStep : inflightInstrumentedSteps) {
    if (blockUntilComplete || runtime->queryEvent(inflightStep.completionEvent)) {
      if (blockUntilComplete) {
        runtime->synchronizeStream(inflightStep.copyStream);
      }
      for (const auto &pendingOp : inflightStep.pendingOps) {
        auto hostOffset = static_cast<size_t>(
            reinterpret_cast<uint64_t>(pendingOp.buffer) -
            inflightStep.stepBufferToken);
        parseCopiedInstrumentedOp(pendingOp, inflightStep.hostBuffer + hostOffset,
                                  pendingOp.size);
        completedBufferPtrs.push_back(
            reinterpret_cast<uint64_t>(pendingOp.buffer));
      }
      releaseHostStagingBuffer(inflightStep.hostBuffer,
                               inflightStep.hostBufferSize);
      runtime->destroyEvent(inflightStep.completionEvent);
    } else {
      pendingInflightSteps.push_back(std::move(inflightStep));
    }
  }
  inflightInstrumentedSteps = std::move(pendingInflightSteps);
}

void InstrumentationProfiler::parseCopiedInstrumentedOp(
    const PendingInstrumentedOp &pendingOp, uint8_t *hostBuffer, size_t size) {
  auto streamId = pendingOp.streamId;
  auto functionId = pendingOp.functionId;
  const auto &functionName = functionNames[functionId];

  auto config = getParserConfig(functionId, size);
  auto circularLayoutConfig =
      std::dynamic_pointer_cast<CircularLayoutParserConfig>(config);
  if (!circularLayoutConfig) {
    throw std::runtime_error(
        "Only circular layout parser is supported for now");
  }

  int64_t timeShiftCost = 0;
  if (modeOptions.count("optimizations")) {
    auto optimizations = proton::split(modeOptions.at("optimizations"), ",");
    if (std::find(optimizations.begin(), optimizations.end(), "time_shift") !=
        optimizations.end())
      timeShiftCost = getTimeShiftCost(*circularLayoutConfig);
  }
  auto &scopeIdContexts = functionScopeIdContexts[functionId];

  ByteSpan byteSpan(hostBuffer, size);
  CircularLayoutParser parser(byteSpan, *circularLayoutConfig);
  parser.parse();
  const auto &blockTraces = parser.getResult()->blockTraces;
  const auto kernelTraceMode = isKernelTraceMode(modeOptions);
  const auto deviceId = pendingOp.deviceId;

  if (kernelTraceMode) {
    // Today we reduce the existing per-CTA timestamps on the host to get
    // one launch interval. If contention on a single per-launch record is
    // too high, we can keep per-CTA records and reduce them later; start
    // with host reduction so we preserve CTA-level detail and avoid an
    // extra device-side reduction pass.
    uint64_t startTime = std::numeric_limits<uint64_t>::max();
    uint64_t endTime = 0;
    bool sawBlockTrace = false;
    for (const auto &blockTrace : blockTraces) {
      startTime = std::min(startTime, blockTrace.initTime);
      endTime = std::max(endTime, blockTrace.postFinalTime);
      sawBlockTrace = true;
    }
    if (sawBlockTrace && endTime >= startTime) {
      for (const auto &[data, baseEntry] : pendingOp.dataToEntryMap) {
        if (dynamic_cast<TraceData *>(data) == nullptr) {
          continue;
        }
        auto entry = data->addOp(baseEntry.phase, baseEntry.id, {});
        entry.upsertMetric(std::make_unique<KernelMetric>(
            startTime, endTime, 1, deviceId,
            static_cast<uint64_t>(runtime->getDeviceType()), streamId));
      }
    }
  }
  for (auto &blockTrace : blockTraces) {
    for (auto &trace : blockTrace.traces) {
      for (auto &event : trace.profileEvents) {
        auto &contexts = scopeIdContexts[event.first->scopeId];
        auto duration = event.second->cycle - event.first->cycle;
        auto normalizedDuration = static_cast<double>(duration) /
                                  (circularLayoutConfig->totalUnits *
                                   circularLayoutConfig->numBlocks);
        for (const auto &[data, baseEntry] : pendingOp.dataToEntryMap) {
          if (kernelTraceMode && dynamic_cast<TraceData *>(data) != nullptr) {
            continue;
          }
          auto kernelId = baseEntry.id;
          auto entry = data->addOp(baseEntry.phase, kernelId, contexts);
          entry.upsertMetric(std::make_unique<CycleMetric>(
              event.first->cycle, event.second->cycle, duration,
              normalizedDuration, kernelId, functionName, blockTrace.blockId,
              blockTrace.procId, trace.uid, deviceId,
              static_cast<uint64_t>(runtime->getDeviceType()), timeShiftCost,
              blockTrace.initTime, blockTrace.preFinalTime,
              blockTrace.postFinalTime));
        }
      }
    }
  }
}

void InstrumentationProfiler::exitInstrumentedOp(uint64_t streamId,
                                                 uint64_t functionId,
                                                 uint8_t *buffer, size_t size) {
  if (!buffer)
    return;

  const auto &functionName = functionNames[functionId];
  enterOp(Scope(functionName));
  auto launchDataEntries = dataToEntryMap;
  exitOp(Scope(functionName));
  if (launchDataEntries.empty()) {
    return;
  }

  void *device = runtime->getDevice();
  void *&copyStream = deviceStreams[device];
  if (!copyStream) {
    copyStream = runtime->getPriorityStream();
  }

  pendingInstrumentedOps.push_back(
      PendingInstrumentedOp{streamId, functionId, buffer, size, currentStepId,
                            static_cast<uint64_t>(
                                reinterpret_cast<uintptr_t>(device)),
                            std::move(launchDataEntries)});
}

void InstrumentationProfiler::markStep(uint64_t streamId,
                                       uint64_t stepBufferToken) {
  void *device = runtime->getDevice();
  void *&copyStream = deviceStreams[device];
  if (!copyStream) {
    copyStream = runtime->getPriorityStream();
  }
  auto *completionEvent = runtime->createEvent();
  runtime->recordEvent(completionEvent, reinterpret_cast<void *>(streamId));
  pendingStepFences.push_back(
      PendingStepFence{currentStepId, stepBufferToken, copyStream,
                       completionEvent});
  ++currentStepId;
}

void InstrumentationProfiler::waitStepBuffer(uint64_t streamId,
                                             uint64_t stepBufferToken) {
  for (const auto &stepFence : pendingStepFences) {
    if (stepFence.stepBufferToken == stepBufferToken) {
      throw std::runtime_error(
          "Profiling step buffer is still pending flush; call proton.flush() "
          "before reusing the slot");
    }
  }
  for (const auto &inflightStep : inflightInstrumentedSteps) {
    if (inflightStep.stepBufferToken == stepBufferToken) {
      runtime->waitEvent(reinterpret_cast<void *>(streamId),
                         inflightStep.completionEvent);
      return;
    }
  }
}

void InstrumentationProfiler::doAddMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &scalarMetrics,
    const std::map<std::string, TensorMetric> &tensorMetrics) {
  if (dataToEntryMap.empty()) {
    for (auto *data : dataSet) {
      data->addMetrics(scopeId, scalarMetrics);
    }
  } else {
    for (const auto &entryIt : dataToEntryMap) {
      const auto &entry = entryIt.second;
      entry.upsertFlexibleMetrics(scalarMetrics);
    }
  }
  // TODO(Keren): handle tensor metrics by making metricBuffer a member of the
  // parent Profiler
}

} // namespace proton
