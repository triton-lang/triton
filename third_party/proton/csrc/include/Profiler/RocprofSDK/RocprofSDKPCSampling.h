#ifndef PROTON_PROFILER_ROCPROFSDK_PC_SAMPLING_H_
#define PROTON_PROFILER_ROCPROFSDK_PC_SAMPLING_H_

#include "Data/Data.h"
#include "Utility/Map.h"

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
#include <unordered_map>
#include <unordered_set>
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

class RocprofSDKPCSampling {
public:
  static constexpr const char *UnknownKernelName = "<unknown>";

  RocprofSDKPCSampling();
  ~RocprofSDKPCSampling();

  void configure(rocprofiler_buffer_tracing_cb_t callback);

  void setEnabled(bool enabled) { pcSamplingEnabled = enabled; }
  bool isEnabled() const { return pcSamplingEnabled; }
  bool isConfigured() const { return pcSamplingConfigured; }
  bool isStarted() const { return pcSamplingStarted; }

  void start();
  void stop();
  void stopNoThrow();
  void flushBuffers();
  void flushBuffersNoThrow();
  void warnIfInvalidInterval();
  void warnIfSourceLocationsUnavailable();

  void recordCodeObjectLoad(
      const rocprofiler_callback_tracing_code_object_load_data_t &load);
  void recordCodeObjectUnload(uint64_t codeObjectId);
  void recordKernelSymbol(
      const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
          &symbol);

  void recordTarget(uint64_t dispatchId, uint64_t kernelId,
                    const DataToEntryMap &dataToEntry, bool needsKernelChild);
  void processBuffer(rocprofiler_record_header_t **headers, size_t numHeaders,
                     uint64_t dropCount);
  void flushAccum();

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

private:
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
  using DispatchToPCSamplingTargetMap =
      ThreadSafeMap<uint64_t, PCSamplingTarget,
                    std::unordered_map<uint64_t, PCSamplingTarget>>;
  using KernelSymbolMap =
      ThreadSafeMap<uint64_t, KernelSymbolInfo,
                    std::unordered_map<uint64_t, KernelSymbolInfo>>;
  using CodeObjectMap =
      ThreadSafeMap<uint64_t, CodeObjectInfo,
                    std::unordered_map<uint64_t, CodeObjectInfo>>;

  static std::unique_ptr<PCSamplingMetric>
  makePCSamplingMetric(const PCSamplingAccum &accum);

  void accumulate(PCSamplingMetric::PCSamplingMetricKind stallKind,
                  bool isStalled, uint64_t dispatchId, uint64_t codeObjectId,
                  uint64_t pcOffset);

  std::optional<SourceLocation>
  resolveSourceLocation(uint64_t codeObjectId, uint64_t pcOffset,
                        const PCSamplingTarget &target);
  bool ensureSourceLocationDecoder(uint64_t codeObjectId);
  void clearSourceLocationCache(uint64_t codeObjectId);
  void releaseUnloadedCodeObject(uint64_t codeObjectId);
  void removeSourceLocationDecoder(const CodeObjectInfo &info);

  bool pcSamplingEnabled{false};
  bool pcSamplingConfigured{false};
  bool pcSamplingStarted{false};
  bool intervalWarningEmitted{false};
  bool sourceLocationWarningEmitted{false};
  uint64_t pcSamplingInterval{1ULL << 17};
  std::string invalidPCSamplingInterval;
  rocprofiler_context_id_t pcSamplingContext{};
  std::vector<rocprofiler_buffer_id_t> pcSamplingBuffers;

  std::mutex pcSamplingMutex;
  PCSamplingAccumMap pcSamplingAccum;
  std::unordered_set<uint64_t> flushingCodeObjectIds;
  std::mutex sourceLocationMutex;
  std::map<SourceLocationKey, std::optional<SourceLocation>>
      sourceLocationCache;
  std::mutex sourceLocationTranslatorMutex;
#if PROTON_ROCPROFILER_SDK_HAS_PC_SAMPLING &&                                  \
    PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE
  std::unique_ptr<
      ::rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate>
      sourceLocationTranslator;
#endif
  CodeObjectMap codeObjects;
  KernelSymbolMap kernelSymbols;
  DispatchToPCSamplingTargetMap dispatchToPCSamplingTarget;
};

} // namespace proton

#endif // PROTON_PROFILER_ROCPROFSDK_PC_SAMPLING_H_
