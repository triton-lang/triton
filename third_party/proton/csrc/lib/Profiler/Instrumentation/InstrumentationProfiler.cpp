#include "Profiler/Instrumentation/InstrumentationProfiler.h"
#include "TraceDataIO/CircularLayoutParser.h"

#include "Driver/GPU/CudaApi.h"
#include "Runtime/CudaRuntime.h"
#include "Runtime/HipRuntime.h"
#include "Utility/Numeric.h"
#include "Utility/String.h"
#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>

namespace proton {

constexpr size_t DEFAULT_HOST_BUFFER_SIZE = 64 * 1024 * 1024;           // 64MB
constexpr size_t MAX_HOST_BUFFER_SIZE = 4LL * 1024LL * 1024LL * 1024LL; // 4GB

thread_local std::map<Data *, size_t> InstrumentationProfiler::dataScopeIdMap =
    std::map<Data *, size_t>(); // Initialize the static member variable

InstrumentationProfiler::~InstrumentationProfiler() {}

void InstrumentationProfiler::doStart() {
  // Start the instrumentation profiler.
}

void InstrumentationProfiler::doFlush() {
  // Flush the instrumentation profiler.
}

void InstrumentationProfiler::doStop() {
  // Stop the instrumentation profiler.
  // FIXME: Also we should ensure the context is valid before releasing the
  // memory
  if (hostBuffer != nullptr) {
    runtime->freeHostBuffer(hostBuffer);
    hostBuffer = nullptr;
  }
  for (auto &[device, deviceStream] : deviceStreams) {
    runtime->destroyStream(deviceStream);
  }
  deviceStreams.clear();
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

void InstrumentationProfiler::enterInstrumentedOp(uint64_t streamId,
                                                  uint64_t functionId,
                                                  uint8_t *buffer,
                                                  size_t size) {
  if (!hostBuffer) {
    runtime->allocateHostBuffer(&hostBuffer, DEFAULT_HOST_BUFFER_SIZE);
  }
}

void InstrumentationProfiler::exitInstrumentedOp(uint64_t streamId,
                                                 uint64_t functionId,
                                                 uint8_t *buffer, size_t size) {
  if (!buffer || !hostBuffer)
    return;

  void *device = runtime->getDevice();
  void *&priorityStream = deviceStreams[device];
  if (!priorityStream) {
    priorityStream = runtime->getPriorityStream();
  }

  if (size > MAX_HOST_BUFFER_SIZE) {
    throw std::runtime_error(
        "Buffer size " + std::to_string(size) + " exceeds the limit " +
        std::to_string(MAX_HOST_BUFFER_SIZE) + ", not supported yet in proton");
  } else if (size > DEFAULT_HOST_BUFFER_SIZE) {
    runtime->freeHostBuffer(hostBuffer);
    auto newSize = nextPowerOfTwo(size);
    runtime->allocateHostBuffer(&hostBuffer, newSize);
  }

  auto dataSet = getDataSet();
  const auto &functionName = functionNames[functionId];
  if (dataScopeIdMap.empty()) {
    for (auto &data : dataSet) {
      auto scopeId = Scope::getNewScopeId();
      data->addOp(scopeId, functionName);
      dataScopeIdMap[data] = scopeId;
    }
  }

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

  runtime->synchronizeStream(reinterpret_cast<void *>(streamId));
  runtime->processHostBuffer(
      hostBuffer, size, buffer, size, priorityStream,
      [&](uint8_t *bufferPtr, size_t size) {
        ByteSpan byteSpan(bufferPtr, size);
        CircularLayoutParser parser(byteSpan, *circularLayoutConfig);
        parser.parse();
        for (auto &blockTrace : parser.getResult()->blockTraces) {
          for (auto &trace : blockTrace.traces) {
            for (auto &event : trace.profileEvents) {
              auto &contexts = scopeIdContexts[event.first->scopeId];
              auto duration = event.second->cycle - event.first->cycle;
              auto normalizedDuration = static_cast<double>(duration) /
                                        (circularLayoutConfig->totalUnits *
                                         circularLayoutConfig->numBlocks);
              for (auto *data : dataSet) {
                auto kernelId = dataScopeIdMap[data];
                auto scopeId = data->addOp(kernelId, contexts);
                data->addMetric(
                    scopeId,
                    std::make_shared<CycleMetric>(
                        event.first->cycle, event.second->cycle, duration,
                        normalizedDuration, kernelId, functionName,
                        blockTrace.blockId, blockTrace.procId, trace.uid,
                        static_cast<uint64_t>(
                            reinterpret_cast<uintptr_t>(device)),
                        static_cast<uint64_t>(runtime->getDeviceType()),
                        timeShiftCost, blockTrace.initTime,
                        blockTrace.preFinalTime, blockTrace.postFinalTime));
              }
            }
          }
        }
      });

  dataScopeIdMap.clear();
}

void InstrumentationProfiler::doAddMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &scalarMetrics,
    const std::map<std::string, TensorMetric> &tensorMetrics) {
  // Currently no-op
  for (auto *data : getDataSet()) {
    data->addMetrics(scopeId, scalarMetrics);
  }
  // TODO(Keren): handle tensor metrics by making metricBuffer a member of the
  // parent Profiler
}

} // namespace proton
