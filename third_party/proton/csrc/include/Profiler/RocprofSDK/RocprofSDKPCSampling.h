#ifndef PROTON_PROFILER_ROCPROFSDK_PC_SAMPLING_H_
#define PROTON_PROFILER_ROCPROFSDK_PC_SAMPLING_H_

#include "Data/Data.h"

#include "rocprofiler-sdk/buffer.h"
#include "rocprofiler-sdk/callback_tracing.h"
#include "rocprofiler-sdk/fwd.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if PROTON_ROCPROFILER_SDK_HAS_PC_SAMPLING &&                                  \
    PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE
namespace rocprofiler {
namespace sdk {
namespace codeobj {
namespace disassembly {
class CodeobjAddressTranslate;
} // namespace disassembly
} // namespace codeobj
} // namespace sdk
} // namespace rocprofiler
#endif

namespace proton {

// Owns the AMD PC-sampling service and its asynchronous attribution pipeline.
// configure() creates the SDK context, buffers, and callback threads;
// start() activates sampling only for pcsampling mode. Code-object, symbol, and
// dispatch callbacks may run concurrently with sample-buffer processing.
// High-frequency sample bookkeeping and attribution metadata are kept in two
// separate lock-bound state objects so source decoding cannot block sampling.
//
// Code-object images remain available after an unload notification because
// source locations are resolved asynchronously after samples are delivered.
// They are released after both the active accumulator and the flushing
// snapshot have drained. The owner stops sampling and flushes both SDK buffers
// and the accumulator before destruction; an unsuccessful configure leaves
// start(), stop(), and flushing as no-ops.
class RocprofSDKPCSampling {
public:
  static constexpr const char *UnknownKernelName = "<unknown>";

  RocprofSDKPCSampling();
  ~RocprofSDKPCSampling();

  void configure(rocprofiler_buffer_tracing_cb_t callback);

  bool isConfigured() const { return pcSamplingServiceConfigured; }
  const std::string &configurationFailureReason() const {
    return pcSamplingConfigurationFailureReason;
  }
  bool isStarted() const { return pcSamplingStarted; }

  void start(bool pcSamplingModeEnabled);
  void stop();
  void stopNoThrow();
  void flushBuffers();
  void flushBuffersNoThrow();
  void warnIfInvalidInterval();
  void warnIfSourceLocationsUnavailable();

  void recordCodeObjectLoad(
      const rocprofiler_callback_tracing_code_object_load_data_t &load,
      bool pcSamplingModeEnabled);
  void recordCodeObjectUnload(uint64_t codeObjectId);
  void recordKernelSymbol(
      const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
          &symbol);

  template <typename CorrIdToExternIdMap, typename ExternIdToStateMap>
  void recordTarget(uint64_t dispatchId, uint64_t correlationId,
                    uint64_t kernelId, size_t graphExternId,
                    CorrIdToExternIdMap &corrIdToExternId,
                    ExternIdToStateMap &externIdToState) {
    if (dispatchId == 0)
      return;

    size_t externId = graphExternId;
    if (externId == Scope::DummyScopeId &&
        !corrIdToExternId.withRead(
            correlationId, [&](const size_t &value) { externId = value; }))
      return;
    if (externId == Scope::DummyScopeId)
      return;

    DataToEntryMap dataToEntry;
    bool needsKernelChild = false;
    if (!externIdToState.withRead(externId, [&](const auto &state) {
          dataToEntry = state.dataToEntry;
          needsKernelChild =
              graphExternId != Scope::DummyScopeId || state.isMissingName;
        }))
      return;
    recordResolvedTarget(dispatchId, kernelId, dataToEntry, needsKernelChild);
  }
  void processBuffer(rocprofiler_record_header_t **headers, size_t numHeaders,
                     uint64_t dropCount);
  void flushAccum();

private:
  struct PCSamplingAccum {
    uint64_t values[PCSamplingMetric::PCSamplingMetricKind::Count] = {};
  };

  struct PCSamplingTarget {
    std::string kernelName = UnknownKernelName;
    uint64_t codeObjectId{0};
    DataToEntryMap dataToEntry;
    bool needsKernelChild{true};
  };

  struct CodeObjectInfo {
    uint64_t codeObjectId{0};
    uint64_t loadSize{0};
    int64_t loadDelta{0};
    std::vector<char> image;
    bool unloaded{false};
    bool decoderRegistered{false};
  };

  struct KernelSymbolInfo {
    std::string name = UnknownKernelName;
    uint64_t codeObjectId{0};
  };

  struct SourceLocation {
    std::string file;
    uint32_t line{0};
    std::string function;
  };

  struct SourceLocationKey {
    uint64_t codeObjectId{0};
    uint64_t pcOffset{0};

    bool operator<(const SourceLocationKey &other) const {
      return codeObjectId < other.codeObjectId ||
             (codeObjectId == other.codeObjectId && pcOffset < other.pcOffset);
    }
  };

  struct PCSamplingKey {
    uint64_t dispatchId{0};
    uint64_t codeObjectId{0};
    uint64_t pcOffset{0};

    bool operator==(const PCSamplingKey &other) const {
      return dispatchId == other.dispatchId &&
             codeObjectId == other.codeObjectId && pcOffset == other.pcOffset;
    }
  };

  struct PCSamplingKeyHash {
    size_t operator()(const PCSamplingKey &key) const {
      constexpr uint64_t HashCombineConstant = 0x9e3779b97f4a7c15ULL;
      size_t seed = std::hash<uint64_t>{}(key.dispatchId);
      seed ^= std::hash<uint64_t>{}(key.codeObjectId) + HashCombineConstant +
              (seed << 6) + (seed >> 2);
      seed ^= std::hash<uint64_t>{}(key.pcOffset) + HashCombineConstant +
              (seed << 6) + (seed >> 2);
      return seed;
    }
  };

  using PCSamplingAccumMap =
      std::unordered_map<PCSamplingKey, PCSamplingAccum, PCSamplingKeyHash>;

  struct SamplingState {
    PCSamplingAccumMap accum;
    // Code objects represented in accum, maintained to avoid scanning accum
    // while holding the sampling lock.
    std::unordered_set<uint64_t> pendingCodeObjectIds;
    std::unordered_set<uint64_t> flushingCodeObjectIds;
  };

  struct MetadataState {
    MetadataState();
    ~MetadataState();

    std::map<SourceLocationKey, std::optional<SourceLocation>>
        sourceLocationCache;
#if PROTON_ROCPROFILER_SDK_HAS_PC_SAMPLING &&                                  \
    PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE
    std::unique_ptr<
        ::rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate>
        sourceLocationTranslator;
#endif
    std::unordered_map<uint64_t, CodeObjectInfo> codeObjects;
    std::unordered_set<uint64_t> sourceLocationDiagnosticEmitted;
    std::unordered_map<uint64_t, KernelSymbolInfo> kernelSymbols;
    std::unordered_map<uint64_t, PCSamplingTarget> dispatchTargets;
  };

  // Couples a state object to the only mutex that may protect it. State is
  // available exclusively inside withLock() callbacks, and references cannot
  // escape those callbacks.
  template <typename State> class LockedState {
  public:
    template <typename Fn> decltype(auto) withLock(Fn &&fn) {
      using Result = std::invoke_result_t<Fn, State &>;
      static_assert(!std::is_reference_v<Result>,
                    "Locked state references must not escape");
      std::lock_guard<std::mutex> lock(mutex);
      return std::forward<Fn>(fn)(state);
    }

  private:
    std::mutex mutex;
    State state;
  };

  static std::unique_ptr<PCSamplingMetric>
  makePCSamplingMetric(const PCSamplingAccum &accum);
  static std::optional<SourceLocation>
  parseSourceLocationComment(const std::string &comment,
                             const std::string &fallbackFunction);
  void recordResolvedTarget(uint64_t dispatchId, uint64_t kernelId,
                            const DataToEntryMap &dataToEntry,
                            bool needsKernelChild);

  void accumulate(PCSamplingMetric::PCSamplingMetricKind stallKind,
                  bool isStalled, uint64_t dispatchId, uint64_t codeObjectId,
                  uint64_t pcOffset);

  std::optional<SourceLocation>
  resolveSourceLocationLocked(MetadataState &state, uint64_t codeObjectId,
                              uint64_t pcOffset,
                              const PCSamplingTarget &target);
  bool ensureSourceLocationDecoderLocked(MetadataState &state,
                                         uint64_t codeObjectId);
  void clearSourceLocationCacheLocked(MetadataState &state,
                                      uint64_t codeObjectId);
  void replaceCodeObjectLocked(MetadataState &state, CodeObjectInfo info);
  void removeSourceLocationDecoderLocked(MetadataState &state,
                                         const CodeObjectInfo &info);
  void reportSourceLocationErrorLocked(MetadataState &state,
                                       uint64_t codeObjectId,
                                       const char *operation,
                                       const char *detail = nullptr);
  void tryReleaseCodeObject(uint64_t codeObjectId);

  // Set when rocprofiler_force_configure successfully configures the service
  // for at least one GPU agent.
  bool pcSamplingServiceConfigured{false};
  bool pcSamplingStarted{false};
  bool intervalWarningEmitted{false};
  bool sourceLocationWarningEmitted{false};
  uint64_t pcSamplingInterval{1ULL << 17};
  std::string invalidPCSamplingInterval;
  std::string pcSamplingConfigurationFailureReason{
      "Proton was built without rocprofiler-sdk PC sampling support"};
  rocprofiler_context_id_t pcSamplingContext{};
  std::vector<rocprofiler_buffer_id_t> pcSamplingBuffers;

  // A flush consumes dispatch targets, so concurrent flushes cannot process
  // independent snapshots safely.
  std::mutex flushMutex;
  // This state is touched by high-frequency buffer callbacks. Keep it separate
  // from metadata so DWARF decoding never blocks sample accumulation.
  LockedState<SamplingState> samplingState;
  // Code objects, decoders, source cache entries, symbols, and dispatch targets
  // form one attribution domain and are always accessed under this lock.
  // If both state locks are needed, acquire metadata before sampling.
  // This order ensures a thread waiting on slower metadata work does not block
  // sample ingestion, but must be enforced consistently to prevent deadlock.
  LockedState<MetadataState> metadataState;
};

} // namespace proton

#endif // PROTON_PROFILER_ROCPROFSDK_PC_SAMPLING_H_
