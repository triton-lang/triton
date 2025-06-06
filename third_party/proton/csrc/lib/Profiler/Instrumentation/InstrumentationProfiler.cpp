#include "Profiler/Instrumentation/InstrumentationProfiler.h"
#include "TraceDataIO/CircularLayoutParser.h"

#include "Driver/GPU/CudaApi.h"
#include "Profiler/Instrumentation/CudaRuntime.h"
#include "Profiler/Instrumentation/HipRuntime.h"
#include "Utility/String.h"
#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>

namespace proton {

constexpr size_t HOST_BUFFER_SIZE = 256 * 1024 * 1024;

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
}

InstrumentationProfiler *
InstrumentationProfiler::setMode(const std::vector<std::string> &mode) {
  if (mode.empty()) {
    throw std::runtime_error("Mode cannot be empty");
  }
  if (toLower(mode[0]) == toLower(DeviceTraits<DeviceType::CUDA>::name)) {
    runtime = std::make_unique<CudaRuntime>();
  } else if (toLower(mode[0]) == toLower(DeviceTraits<DeviceType::HIP>::name)) {
    runtime = std::make_unique<HipRuntime>();
  } else {
    throw std::runtime_error("Unknown device type: " + mode[0]);
  }
  for (size_t i = 1; i < mode.size(); ++i) {
    auto delimiterPos = mode[i].find('=');
    if (delimiterPos != std::string::npos) {
      std::string key = mode[i].substr(0, delimiterPos);
      std::string value = mode[i].substr(delimiterPos + 1);
      modeOptions[key] = value;
    } else {
      modeOptions[mode[i]] = "";
    }
  }

  return this;
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
    runtime->allocateHostBuffer(&hostBuffer, HOST_BUFFER_SIZE);
  }
}

void InstrumentationProfiler::exitInstrumentedOp(uint64_t streamId,
                                                 uint64_t functionId,
                                                 uint8_t *buffer, size_t size) {
  if (!buffer || !hostBuffer)
    return;
  uint64_t device = runtime->getDevice();
  void *&priorityStream = deviceStreams[reinterpret_cast<void *>(device)];
  if (!priorityStream) {
    priorityStream = runtime->getPriorityStream();
  }
  // FIXME: we should support cases where a single host buffer contains multiple
  // device buffer or a single device buffer contains multiple host buffer
  if (size > HOST_BUFFER_SIZE) {
    throw std::runtime_error(
        "Buffer size " + std::to_string(size) + " exceeds the limit " +
        std::to_string(HOST_BUFFER_SIZE) + ", not supported yet in proton");
  }

  auto dataSet = getDataSet();

  std::string functionName = functionNames[functionId];
  if (dataScopeIdMap.empty()) {
    for (auto &data : dataSet) {
      auto scopeId = Scope::getNewScopeId();
      data->addOp(scopeId, functionName);
      dataScopeIdMap[data] = scopeId;
    }
  }

  // For now, only support the synchronization mode
  runtime->synchronizeStream(reinterpret_cast<void *>(streamId));

  CircularLayoutParserConfig config;
  config.scratchMemSize =
      functionMetadata.at(functionId).getScratchMemorySize();
  if (!(modeOptions.count("granularity") == 0 ||
        modeOptions.at("granularity") == "GRANULARITY.WARP")) {
    throw std::runtime_error("Only warp granularity is supported for now");
  }
  config.totalUnits = functionMetadata.at(functionId).getNumWarps();
  config.numBlocks = size / config.scratchMemSize;
  config.uidVec = getUnitIdVector(modeOptions, config.totalUnits);
  config.device = Device();
  config.device.type = runtime->getDeviceType();

  uint32_t timeShiftCost = 0;
  if (modeOptions.count("optimization") &&
      modeOptions.at("optimization") == "time_shift")
    timeShiftCost = getTimeShiftCost(config);

  auto &scopeIdContexts = functionScopeIdContexts[functionId];

  runtime->processHostBuffer(
      hostBuffer, HOST_BUFFER_SIZE, buffer, size, priorityStream,
      [&](uint8_t *bufferPtr, size_t size) {
        auto byteSpan = ByteSpan(bufferPtr, size);
        auto parser = CircularLayoutParser(byteSpan, config);
        parser.parse();
        for (auto &blockTrace : parser.getResult()->blockTraces) {
          for (auto &trace : blockTrace.traces) {
            auto &profileEvents = trace.profileEvents;
            for (auto &event : profileEvents) {
              // Process the profile events
              auto &contexts = scopeIdContexts[event.first->scopeId];
              auto duration = event.second->cycle - event.first->cycle;
              auto normalizedDuration = static_cast<double>(duration) /
                                        (config.totalUnits * config.numBlocks);
              for (auto *data : dataSet) {
                auto kernelId = dataScopeIdMap[data];
                auto scopeId = data->addOp(dataScopeIdMap[data], contexts);
                data->addMetric(
                    scopeId,
                    std::make_shared<CycleMetric>(
                        event.first->cycle, event.second->cycle, duration,
                        normalizedDuration, kernelId, functionName,
                        blockTrace.blockId, blockTrace.procId, trace.uid,
                        device, static_cast<uint64_t>(runtime->getDeviceType()),
                        timeShiftCost));
              }
            }
          }
        }
      });

  dataScopeIdMap.clear();
}

} // namespace proton
