#include "Profiler/RocprofSDK/RocprofSDKPCSampling.h"

#if PROTON_ROCPROFILER_SDK_HAS_PC_SAMPLING

#include "Context/Context.h"
#include "Driver/GPU/RocprofApi.h"
#include "Utility/Env.h"

#include "rocprofiler-sdk/agent.h"
#include "rocprofiler-sdk/pc_sampling.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <system_error>
#include <unordered_set>
#include <utility>

namespace proton {

namespace {

constexpr size_t PCSamplingBufferSize = 1024 * 1024;
struct AgentPCSamplingConfig {
  rocprofiler_agent_id_t agentId;
  std::vector<rocprofiler_pc_sampling_configuration_t> configs;
};

rocprofiler_status_t
pcSamplingConfigCallback(const rocprofiler_pc_sampling_configuration_t *configs,
                         size_t numConfigs, void *userData) {
  auto *out =
      static_cast<std::vector<rocprofiler_pc_sampling_configuration_t> *>(
          userData);
  for (size_t i = 0; i < numConfigs; ++i)
    out->push_back(configs[i]);
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t agentQueryCallback(rocprofiler_agent_version_t version,
                                        const void **agents, size_t count,
                                        void *userData) {
  if (version != ROCPROFILER_AGENT_INFO_VERSION_0)
    return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

  auto *out = static_cast<std::vector<AgentPCSamplingConfig> *>(userData);
  auto agentList = reinterpret_cast<const rocprofiler_agent_t *const *>(agents);
  for (size_t i = 0; i < count; ++i) {
    if (agentList[i]->type != ROCPROFILER_AGENT_TYPE_GPU)
      continue;

    AgentPCSamplingConfig entry{agentList[i]->id, {}};
    auto status = rocprofiler::queryPCSamplingAgentConfigurations<false>(
        agentList[i]->id, pcSamplingConfigCallback, &entry.configs);
    if (status == ROCPROFILER_STATUS_SUCCESS && !entry.configs.empty())
      out->push_back(std::move(entry));
  }
  return ROCPROFILER_STATUS_SUCCESS;
}

const rocprofiler_pc_sampling_configuration_t *pickPCSamplingConfig(
    const std::vector<rocprofiler_pc_sampling_configuration_t> &configs,
    std::optional<rocprofiler_pc_sampling_method_t> requestedMethod) {
  if (requestedMethod) {
    auto requested =
        std::find_if(configs.begin(), configs.end(), [&](const auto &cfg) {
          return cfg.method == *requestedMethod;
        });
    return requested == configs.end() ? nullptr : &*requested;
  }

  auto stochastic =
      std::find_if(configs.begin(), configs.end(), [](const auto &cfg) {
        return cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC;
      });
  if (stochastic != configs.end())
    return &*stochastic;

  auto hostTrap =
      std::find_if(configs.begin(), configs.end(), [](const auto &cfg) {
        return cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;
      });
  if (hostTrap != configs.end())
    return &*hostTrap;
  return nullptr;
}

bool parsePCSamplingMethod(
    const std::string &value,
    std::optional<rocprofiler_pc_sampling_method_t> &method) {
  if (value.empty()) {
    method = std::nullopt;
    return true;
  }
  if (value == "stochastic") {
    method = ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC;
    return true;
  }
  if (value == "host-trap") {
    method = ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;
    return true;
  }
  return false;
}

PCSamplingMetric::PCSamplingMetricKind mapNotIssuedReasonToStallMetric(
    rocprofiler_pc_sampling_instruction_not_issued_reason_t reason) {
  constexpr std::array ReasonToMetric = {
      PCSamplingMetric::StalledMisc,
      PCSamplingMetric::StalledNoInstruction,
      PCSamplingMetric::StalledAMDALUDependency,
      PCSamplingMetric::StalledAMDWaitcnt,
      PCSamplingMetric::StalledAMDInternalInstruction,
      PCSamplingMetric::StalledBarrier,
      PCSamplingMetric::StalledNotSelected,
      PCSamplingMetric::StalledAMDArbiterWinExStall,
      PCSamplingMetric::StalledAMDOtherWait,
      PCSamplingMetric::StalledSleeping,
  };

  auto index = static_cast<int64_t>(reason);
  if (index < 0 || static_cast<size_t>(index) >= ReasonToMetric.size())
    return PCSamplingMetric::StalledMisc;
  return ReasonToMetric[index];
}

template <bool CheckSuccess>
void stopContextIfStarted(rocprofiler_context_id_t context, bool &started) {
  if (!started)
    return;
  rocprofiler::stopContext<CheckSuccess>(context);
  started = false;
}

template <bool CheckSuccess>
void flushBuffers(const std::vector<rocprofiler_buffer_id_t> &buffers) {
  for (auto &buffer : buffers)
    rocprofiler::flushBuffer<CheckSuccess>(buffer);
}

template <typename SampleT>
rocprofiler_pc_t getSamplePC(const SampleT *sample) {
  if (sample->size >= offsetof(SampleT, pc) + sizeof(sample->pc))
    return sample->pc;
  return rocprofiler_pc_t{ROCPROFILER_CODE_OBJECT_ID_NONE, 0};
}

template <typename SampleT>
bool hasSampleField(const SampleT *sample, size_t offset, size_t size) {
  return sample->size >= offset + size;
}

std::optional<uint64_t> parsePCSamplingInterval(const std::string &value) {
  if (value.empty())
    return std::nullopt;

  uint64_t parsed = 0;
  auto begin = value.data();
  auto end = begin + value.size();
  auto [ptr, ec] = std::from_chars(begin, end, parsed);
  if (ec != std::errc{} || ptr != end || parsed == 0)
    return std::nullopt;
  return parsed;
}

const char *rocprofilerStatusName(rocprofiler_status_t status) {
  switch (status) {
  case ROCPROFILER_STATUS_SUCCESS:
    return "ROCPROFILER_STATUS_SUCCESS";
  case ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE:
    return "ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE";
  default:
    return "ROCPROFILER_STATUS_ERROR";
  }
}

} // namespace

RocprofSDKPCSampling::RocprofSDKPCSampling() = default;

RocprofSDKPCSampling::~RocprofSDKPCSampling() = default;

void RocprofSDKPCSampling::configure(rocprofiler_buffer_tracing_cb_t callback) {
  pcSamplingConfigurationFailureReason.clear();
  auto methodStr = getStrEnv("PROTON_ROCPROFILER_PC_SAMPLING_METHOD");
  std::optional<rocprofiler_pc_sampling_method_t> requestedMethod;
  if (!parsePCSamplingMethod(methodStr, requestedMethod)) {
    pcSamplingConfigurationFailureReason =
        "invalid PROTON_ROCPROFILER_PC_SAMPLING_METHOD='" + methodStr +
        "'; expected 'stochastic' or 'host-trap'";
    return;
  }
  auto intervalStr = getStrEnv("PROTON_PC_SAMPLING_INTERVAL");
  if (!intervalStr.empty()) {
    auto parsedInterval = parsePCSamplingInterval(intervalStr);
    if (parsedInterval) {
      pcSamplingInterval = *parsedInterval;
    } else {
      invalidPCSamplingInterval = intervalStr;
    }
  }
  rocprofiler::createContext<true>(&pcSamplingContext);

  std::vector<AgentPCSamplingConfig> agentsWithPCSampling;
  rocprofiler::queryAvailableAgents<true>(
      ROCPROFILER_AGENT_INFO_VERSION_0, agentQueryCallback,
      sizeof(rocprofiler_agent_t), &agentsWithPCSampling);

  if (agentsWithPCSampling.empty()) {
    pcSamplingConfigurationFailureReason =
        "rocprofiler-sdk did not report PC sampling configurations for any "
        "visible AMD GPU agent";
    return;
  }

  std::stringstream failureDetails;
  size_t failedConfigCount = 0;
  size_t unsupportedConfigCount = 0;

  for (auto &agent : agentsWithPCSampling) {
    auto *picked = pickPCSamplingConfig(agent.configs, requestedMethod);
    if (!picked) {
      failureDetails << " agent " << agent.agentId.handle;
      if (requestedMethod)
        failureDetails
            << " does not support PROTON_ROCPROFILER_PC_SAMPLING_METHOD='"
            << methodStr << "';";
      else
        failureDetails << " has no supported PC sampling method;";
      ++failedConfigCount;
      continue;
    }

    auto interval = pcSamplingInterval;
    if (interval < picked->min_interval)
      interval = picked->min_interval;
    if (interval > picked->max_interval)
      interval = picked->max_interval;

    rocprofiler_buffer_id_t pcSamplingBuffer{};
    size_t pcSamplingWatermark =
        PCSamplingBufferSize - (PCSamplingBufferSize / 4);
    rocprofiler::createBuffer<true>(pcSamplingContext, PCSamplingBufferSize,
                                    pcSamplingWatermark,
                                    ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                    callback, nullptr, &pcSamplingBuffer);

    auto cfgStatus = rocprofiler::configurePCSamplingService<false>(
        pcSamplingContext, agent.agentId, picked->method, picked->unit,
        interval, pcSamplingBuffer, 0);

    if (cfgStatus == ROCPROFILER_STATUS_SUCCESS) {
      rocprofiler_callback_thread_t pcSamplingThread{};
      rocprofiler::createCallbackThread<true>(&pcSamplingThread);
      rocprofiler::assignCallbackThread<true>(pcSamplingBuffer,
                                              pcSamplingThread);
      pcSamplingBuffers.push_back(pcSamplingBuffer);
      pcSamplingServiceConfigured = true;
    } else {
      failureDetails << " agent " << agent.agentId.handle
                     << " status=" << rocprofilerStatusName(cfgStatus) << "("
                     << cfgStatus << ");";
      ++failedConfigCount;
      if (cfgStatus == ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE)
        ++unsupportedConfigCount;
    }
    if (cfgStatus != ROCPROFILER_STATUS_SUCCESS &&
        cfgStatus != ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE) {
      std::cerr << "[PROTON] PC sampling configuration failed on agent "
                << agent.agentId.handle << " status=" << cfgStatus << std::endl;
    }
  }

  if (!pcSamplingServiceConfigured) {
    // NOT_AVAILABLE is the expected result for unsupported agents; distinguish
    // that case from unexpected SDK failures while configuring the service.
    if (failedConfigCount == unsupportedConfigCount) {
      pcSamplingConfigurationFailureReason =
          "rocprofiler-sdk PC sampling service is not available for the "
          "visible AMD GPU agents.";
    } else {
      pcSamplingConfigurationFailureReason =
          "rocprofiler-sdk failed to configure PC sampling for the visible AMD "
          "GPU agents.";
    }
    pcSamplingConfigurationFailureReason += failureDetails.str();
  }
}

void RocprofSDKPCSampling::warnIfInvalidInterval() {
  if (invalidPCSamplingInterval.empty() || intervalWarningEmitted)
    return;
  intervalWarningEmitted = true;
  std::cerr << "[PROTON] Ignoring invalid PROTON_PC_SAMPLING_INTERVAL='"
            << invalidPCSamplingInterval
            << "'; expected a positive integer. Using the default interval."
            << std::endl;
}

void RocprofSDKPCSampling::warnIfSourceLocationsUnavailable() {
  if (sourceLocationWarningEmitted ||
      PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE)
    return;
  sourceLocationWarningEmitted = true;
  std::cerr
      << "[PROTON] AMD PC sampling source-line attribution is unavailable "
         "with this rocprofiler-sdk build; samples will fall back to "
         "kernel-level attribution."
      << std::endl;
}

void RocprofSDKPCSampling::recordKernelSymbol(
    const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
        &symbol) {
  if (symbol.kernel_id == 0 || symbol.kernel_name == nullptr)
    return;

  KernelSymbolInfo info;
  info.name = symbol.kernel_name;
  // AMDGPU ELF objects append ".kd" (kernel descriptor) to symbol names.
  // Strip it so user-visible kernel names match the source.
  const std::string suffix = ".kd";
  if (info.name.size() > suffix.size() &&
      info.name.compare(info.name.size() - suffix.size(), suffix.size(),
                        suffix) == 0)
    info.name.resize(info.name.size() - suffix.size());
  info.codeObjectId = symbol.code_object_id;
  metadataState.withLock([&](MetadataState &state) {
    state.kernelSymbols.insert_or_assign(symbol.kernel_id, std::move(info));
  });
}

void RocprofSDKPCSampling::start(bool pcSamplingModeEnabled) {
  if (pcSamplingModeEnabled && pcSamplingServiceConfigured &&
      !pcSamplingStarted) {
    rocprofiler::startContext<true>(pcSamplingContext);
    pcSamplingStarted = true;
  }
}

void RocprofSDKPCSampling::stop() {
  stopContextIfStarted<true>(pcSamplingContext, pcSamplingStarted);
}

void RocprofSDKPCSampling::stopNoThrow() {
  stopContextIfStarted<false>(pcSamplingContext, pcSamplingStarted);
}

void RocprofSDKPCSampling::flushBuffers() {
  proton::flushBuffers<true>(pcSamplingBuffers);
}

void RocprofSDKPCSampling::flushBuffersNoThrow() {
  proton::flushBuffers<false>(pcSamplingBuffers);
}

void RocprofSDKPCSampling::recordResolvedTarget(
    uint64_t dispatchId, uint64_t kernelId, const DataToEntryMap &dataToEntry,
    bool needsKernelChild) {
  PCSamplingTarget target;
  target.dataToEntry = dataToEntry;
  target.needsKernelChild = needsKernelChild;
  metadataState.withLock([&](MetadataState &state) {
    auto symbol = state.kernelSymbols.find(kernelId);
    if (symbol != state.kernelSymbols.end()) {
      target.kernelName = symbol->second.name;
      target.codeObjectId = symbol->second.codeObjectId;
    }
    state.dispatchTargets.insert_or_assign(dispatchId, std::move(target));
  });
}

std::unique_ptr<PCSamplingMetric>
RocprofSDKPCSampling::makePCSamplingMetric(const PCSamplingAccum &accum) {
  auto metric = std::make_unique<PCSamplingMetric>();
  for (int i = 0; i < PCSamplingMetric::PCSamplingMetricKind::Count; ++i)
    metric->updateValue(
        i, MetricValueType(static_cast<uint64_t>(accum.values[i])));
  return metric;
}

void RocprofSDKPCSampling::accumulate(
    PCSamplingMetric::PCSamplingMetricKind stallKind, bool isStalled,
    uint64_t dispatchId, uint64_t codeObjectId, uint64_t pcOffset) {
  samplingState.withLock([&](SamplingState &state) {
    if (codeObjectId != ROCPROFILER_CODE_OBJECT_ID_NONE)
      state.pendingCodeObjectIds.insert(codeObjectId);
    auto &accum = state.accum[{dispatchId, codeObjectId, pcOffset}];
    accum.values[PCSamplingMetric::NumSamples]++;
    if (isStalled) {
      accum.values[PCSamplingMetric::NumStalledSamples]++;
      accum.values[stallKind]++;
    }
  });
}

void RocprofSDKPCSampling::processBuffer(rocprofiler_record_header_t **headers,
                                         size_t numHeaders,
                                         uint64_t dropCount) {
  if (dropCount > 0) {
    std::cerr << "[PROTON] ROCProfiler-SDK dropped " << dropCount
              << " PC sampling records" << std::endl;
  }

  for (size_t i = 0; i < numHeaders; ++i) {
    auto *header = headers[i];
    if (!header || header->category != ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
      continue;

    if (header->kind == ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE) {
      auto *sample =
          static_cast<rocprofiler_pc_sampling_record_stochastic_v0_t *>(
              header->payload);
      using StochasticSample = rocprofiler_pc_sampling_record_stochastic_v0_t;
      bool hasWaveIssueInfo = hasSampleField(
          sample, offsetof(StochasticSample, hw_id), sizeof(sample->hw_id));
      bool hasSnapshot =
          hasSampleField(sample, offsetof(StochasticSample, snapshot),
                         sizeof(sample->snapshot));
      bool isStalled = hasWaveIssueInfo && !sample->wave_issued;
      auto stallKind =
          isStalled && hasSnapshot
              ? mapNotIssuedReasonToStallMetric(
                    static_cast<
                        rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
                        sample->snapshot.reason_not_issued))
              : PCSamplingMetric::StalledSelected;
      auto pc = getSamplePC(sample);
      accumulate(stallKind, isStalled, sample->dispatch_id, pc.code_object_id,
                 pc.code_object_offset);
    } else if (header->kind ==
               ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE) {
      auto *sample =
          static_cast<rocprofiler_pc_sampling_record_host_trap_v0_t *>(
              header->payload);
      auto pc = getSamplePC(sample);
      accumulate(PCSamplingMetric::NumSamples, false, sample->dispatch_id,
                 pc.code_object_id, pc.code_object_offset);
    }
  }
}

void RocprofSDKPCSampling::flushAccum() {
  std::lock_guard<std::mutex> flushLock(flushMutex);
  PCSamplingAccumMap snapshot;
  std::unordered_set<uint64_t> snapshotCodeObjectIds;
  samplingState.withLock([&](SamplingState &state) {
    snapshot.swap(state.accum);
    snapshotCodeObjectIds.swap(state.pendingCodeObjectIds);
    state.flushingCodeObjectIds.insert(snapshotCodeObjectIds.begin(),
                                       snapshotCodeObjectIds.end());
  });
  if (snapshot.empty())
    return;

  std::unordered_map<uint64_t, PCSamplingAccum> unresolvedAccum;
  std::unordered_set<uint64_t> consumedDispatchIds;

  for (auto &[key, accum] : snapshot) {
    const auto dispatchId = key.dispatchId;
    const auto codeObjectId = key.codeObjectId;
    const auto pcOffset = key.pcOffset;
    PCSamplingTarget target;
    std::optional<SourceLocation> sourceLocation;
    bool hasTarget = metadataState.withLock([&](MetadataState &state) {
      auto found = state.dispatchTargets.find(dispatchId);
      if (found == state.dispatchTargets.end())
        return false;
      target = found->second;
      sourceLocation =
          resolveSourceLocationLocked(state, codeObjectId, pcOffset, target);
      return true;
    });

    if (!hasTarget) {
      continue;
    }
    consumedDispatchIds.insert(dispatchId);

    if (!sourceLocation) {
      auto &unresolved = unresolvedAccum[dispatchId];
      for (int i = 0; i < PCSamplingMetric::PCSamplingMetricKind::Count; ++i)
        unresolved.values[i] += accum.values[i];
      continue;
    }

    for (auto &[data, entry] : target.dataToEntry) {
      auto pcEntry = entry;
      if (target.needsKernelChild)
        pcEntry =
            data->addOp(entry.phase, entry.id, {Context(target.kernelName)});
      pcEntry = data->addOp(pcEntry.phase, pcEntry.id,
                            {Context(formatFileLineFunction(
                                sourceLocation->file, sourceLocation->line,
                                sourceLocation->function))});
      pcEntry.upsertMetric(makePCSamplingMetric(accum));
    }
  }

  for (auto &[dispatchId, accum] : unresolvedAccum) {
    const auto currentDispatchId = dispatchId;
    PCSamplingTarget target;
    bool hasTarget = metadataState.withLock([&](MetadataState &state) {
      auto found = state.dispatchTargets.find(currentDispatchId);
      if (found == state.dispatchTargets.end())
        return false;
      target = found->second;
      return true;
    });
    if (!hasTarget)
      continue;

    for (auto &[data, entry] : target.dataToEntry) {
      auto pcEntry = entry;
      if (target.needsKernelChild)
        pcEntry =
            data->addOp(entry.phase, entry.id, {Context(target.kernelName)});
      pcEntry.upsertMetric(makePCSamplingMetric(accum));
    }
  }

  metadataState.withLock([&](MetadataState &state) {
    for (auto dispatchId : consumedDispatchIds)
      state.dispatchTargets.erase(dispatchId);
  });

  samplingState.withLock([&](SamplingState &state) {
    for (auto codeObjectId : snapshotCodeObjectIds)
      state.flushingCodeObjectIds.erase(codeObjectId);
  });

  for (auto codeObjectId : snapshotCodeObjectIds)
    tryReleaseCodeObject(codeObjectId);
}

} // namespace proton

#else

namespace proton {

RocprofSDKPCSampling::RocprofSDKPCSampling() = default;

RocprofSDKPCSampling::~RocprofSDKPCSampling() = default;

RocprofSDKPCSampling::MetadataState::MetadataState() = default;

RocprofSDKPCSampling::MetadataState::~MetadataState() = default;

void RocprofSDKPCSampling::configure(rocprofiler_buffer_tracing_cb_t callback) {
  (void)callback;
}

void RocprofSDKPCSampling::warnIfInvalidInterval() {}

void RocprofSDKPCSampling::warnIfSourceLocationsUnavailable() {}

void RocprofSDKPCSampling::recordCodeObjectLoad(
    const rocprofiler_callback_tracing_code_object_load_data_t &load,
    bool pcSamplingModeEnabled) {
  (void)load;
  (void)pcSamplingModeEnabled;
}

void RocprofSDKPCSampling::recordCodeObjectUnload(uint64_t codeObjectId) {
  (void)codeObjectId;
}

void RocprofSDKPCSampling::recordKernelSymbol(
    const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
        &symbol) {
  (void)symbol;
}

void RocprofSDKPCSampling::start(bool pcSamplingModeEnabled) {
  (void)pcSamplingModeEnabled;
}

void RocprofSDKPCSampling::stop() {}

void RocprofSDKPCSampling::stopNoThrow() {}

void RocprofSDKPCSampling::flushBuffers() {}

void RocprofSDKPCSampling::flushBuffersNoThrow() {}

void RocprofSDKPCSampling::recordResolvedTarget(
    uint64_t dispatchId, uint64_t kernelId, const DataToEntryMap &dataToEntry,
    bool needsKernelChild) {
  (void)dispatchId;
  (void)kernelId;
  (void)dataToEntry;
  (void)needsKernelChild;
}

void RocprofSDKPCSampling::processBuffer(rocprofiler_record_header_t **headers,
                                         size_t numHeaders,
                                         uint64_t dropCount) {
  (void)headers;
  (void)numHeaders;
  (void)dropCount;
}

void RocprofSDKPCSampling::flushAccum() {}

} // namespace proton

#endif
