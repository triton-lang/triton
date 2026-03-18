#include "Profiler/GPUProfiler.h"
#include "Profiler/Graph.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace proton {
namespace detail {

namespace {

struct FlushRange {
  Data *data{nullptr};
  size_t minPhaseToFlush{0};
  size_t maxPhaseToFlush{0};
};

std::pair<std::vector<FlushRange>, std::set<size_t>>
computeFlushRangesAndPeekPhases(
    std::map<Data *, size_t> &dataFlushedPhases,
    const std::map<Data *,
                   std::pair</*start_phase=*/size_t, /*end_phase=*/size_t>>
        &dataPhases,
    const bool peekPendingGraphs) {
  std::vector<FlushRange> flushRanges;
  flushRanges.reserve(dataPhases.size());
  std::set<size_t> phasesToPeek;

  for (auto [data, phase] : dataPhases) {
    if (phase.second == 0) {
      continue;
    }

    auto flushedPhaseIt = dataFlushedPhases.find(data);
    // phase.second at maximum is the current phase, which cannot be a
    // "complete" phase yet. So we flush up to phase.second - 1.
    const size_t endPhaseToFlush = phase.second - 1;

    size_t flushedPhase = Data::kNoCompletePhase;
    if (flushedPhaseIt != dataFlushedPhases.end()) {
      flushedPhase = flushedPhaseIt->second;
    }
    if (data->getBufferedProfileMaxBytes() > 0) {
      const auto serializedUpToPhase =
          data->getBufferedProfileSerializedUpToPhase();
      if (flushedPhase == Data::kNoCompletePhase) {
        flushedPhase = serializedUpToPhase;
      } else if (serializedUpToPhase != Data::kNoCompletePhase) {
        flushedPhase = std::max(flushedPhase, serializedUpToPhase);
      }
    }

    size_t minPhaseToFlush = 0;
    if (flushedPhase == Data::kNoCompletePhase) {
      minPhaseToFlush = 0;
    } else {
      if (endPhaseToFlush <= flushedPhase) {
        continue;
      }
      minPhaseToFlush = flushedPhase + 1;
    }

    flushRanges.push_back(FlushRange{data, minPhaseToFlush, endPhaseToFlush});
    if (peekPendingGraphs) {
      for (size_t p = minPhaseToFlush; p <= endPhaseToFlush; ++p) {
        phasesToPeek.insert(p);
      }
    }
  }

  return {std::move(flushRanges), std::move(phasesToPeek)};
}

struct PeriodicFlushStats {
  uint64_t totalToJsonUs{0};
  uint64_t totalToMsgPackUs{0};
  uint64_t totalJsonWriteUs{0};
  uint64_t totalMsgPackWriteUs{0};
  uint64_t clearUs{0};
  size_t toJsonCalls{0};
  size_t toMsgPackCalls{0};
  size_t jsonWriteCalls{0};
  size_t msgPackWriteCalls{0};
};

void periodicFlushDataPhases(Data &data,
                             const std::string &periodicFlushingFormat,
                             size_t minPhaseToFlush, size_t maxPhaseToFlush,
                             const bool timingEnabled,
                             PeriodicFlushStats &stats) {
  using Clock = std::chrono::steady_clock;
  const auto &path = data.getPath();
  const bool bufferProfiles = data.getBufferedProfileMaxBytes() > 0;
  const bool writeOutputs = !path.empty();

  for (auto startPhase = minPhaseToFlush; startPhase <= maxPhaseToFlush;
       startPhase++) {
    auto pathWithPhase = path + ".part_" + std::to_string(startPhase) + "." +
                         periodicFlushingFormat;

    if (periodicFlushingFormat == "hatchet" ||
        periodicFlushingFormat == "chrome_trace") {
      std::string jsonStr;
      if (timingEnabled) {
        const auto t0 = Clock::now();
        jsonStr = data.toJsonString(startPhase);
        const auto t1 = Clock::now();
        stats.totalToJsonUs +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        ++stats.toJsonCalls;
      } else {
        jsonStr = data.toJsonString(startPhase);
      }

      if (bufferProfiles) {
        data.appendBufferedProfile(startPhase, jsonStr);
      }

      if (!writeOutputs) {
        continue;
      }

      if (timingEnabled) {
        const auto t0 = Clock::now();
        std::ofstream ofs(pathWithPhase, std::ios::out | std::ios::trunc);
        ofs << jsonStr;
        ofs.flush();
        const auto t1 = Clock::now();
        stats.totalJsonWriteUs +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        ++stats.jsonWriteCalls;
      } else {
        std::ofstream ofs(pathWithPhase, std::ios::out | std::ios::trunc);
        ofs << jsonStr;
      }
    } else if (periodicFlushingFormat == "hatchet_msgpack") {
      std::vector<uint8_t> msgPack;
      if (timingEnabled) {
        const auto t0 = Clock::now();
        msgPack = data.toMsgPack(startPhase);
        const auto t1 = Clock::now();
        stats.totalToMsgPackUs +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        ++stats.toMsgPackCalls;
      } else {
        msgPack = data.toMsgPack(startPhase);
      }

      if (bufferProfiles) {
        data.appendBufferedProfile(startPhase, msgPack);
      }

      if (!writeOutputs) {
        continue;
      }

      if (timingEnabled) {
        const auto t0 = Clock::now();
        std::ofstream ofs(pathWithPhase,
                          std::ios::out | std::ios::binary | std::ios::trunc);
        ofs.write(reinterpret_cast<const char *>(msgPack.data()),
                  msgPack.size());
        ofs.flush();
        const auto t1 = Clock::now();
        stats.totalMsgPackWriteUs +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        ++stats.msgPackWriteCalls;
      } else {
        std::ofstream ofs(pathWithPhase,
                          std::ios::out | std::ios::binary | std::ios::trunc);
        ofs.write(reinterpret_cast<const char *>(msgPack.data()),
                  msgPack.size());
      }
    }
  }
}

void periodicClearDataPhases(Data &data, size_t maxPhaseToFlush,
                             const bool timingEnabled,
                             PeriodicFlushStats &stats) {
  using Clock = std::chrono::steady_clock;
  if (!timingEnabled) {
    data.clear(maxPhaseToFlush, /*clearUpToPhase=*/true);
    return;
  }

  const auto t0 = Clock::now();
  data.clear(maxPhaseToFlush, /*clearUpToPhase=*/true);
  const auto t1 = Clock::now();
  stats.clearUs =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

} // namespace

PeriodicFlushingConfig
parsePeriodicFlushingMode(const std::vector<std::string> &modeAndOptions,
                          const char *profilerName) {
  PeriodicFlushingConfig config;
  config.enabled = true;
  for (size_t i = 1; i < modeAndOptions.size(); ++i) {
    const auto delimiterPos = modeAndOptions[i].find('=');
    if (delimiterPos == std::string::npos) {
      throw std::invalid_argument(std::string("[PROTON] ") + profilerName +
                                  ": unsupported option: " +
                                  modeAndOptions[i]);
    }
    const std::string key = modeAndOptions[i].substr(0, delimiterPos);
    const std::string value = modeAndOptions[i].substr(delimiterPos + 1);
    if (key == "format") {
      if (value != "hatchet_msgpack" && value != "chrome_trace" &&
          value != "hatchet") {
        throw std::invalid_argument(std::string("[PROTON] ") + profilerName +
                                    ": unsupported format: " + value);
      }
      config.format = value;
    } else if (key == "buffer_max_bytes") {
      config.bufferMaxBytes = std::stoull(value);
    } else {
      throw std::invalid_argument(std::string("[PROTON] ") + profilerName +
                                  ": unsupported option key: " + key);
    }
  }
  return config;
}

void updateDataPhases(std::map<Data *, std::pair<size_t, size_t>> &dataPhases,
                      Data *data, size_t phase) {
  auto it = dataPhases.find(data);
  if (it == dataPhases.end()) {
    dataPhases.emplace(data, std::make_pair(phase, phase));
  } else {
    it->second.first = std::min(it->second.first, phase);   // start phase
    it->second.second = std::max(it->second.second, phase); // end phase
  }
}

void flushDataPhasesImpl(
    const bool periodicFlushEnabled, const std::string &periodicFlushingFormat,
    std::map<Data *, size_t> &dataFlushedPhases,
    const std::map<Data *,
                   std::pair</*start_phase=*/size_t, /*end_phase=*/size_t>>
        &dataPhases,
    PendingGraphPool *pendingGraphPool) {
  static const bool timingEnabled =
      getBoolEnv("PROTON_DATA_FLUSH_TIMING", false);
  auto [flushRanges, phasesToPeek] = computeFlushRangesAndPeekPhases(
      dataFlushedPhases, dataPhases, pendingGraphPool != nullptr);
  if (pendingGraphPool) {
    using Clock = std::chrono::steady_clock;
    uint64_t totalPeekUs = 0;
    size_t peekCalls = 0;
    for (const auto phase : phasesToPeek) {
      if (timingEnabled) {
        const auto t0 = Clock::now();
        pendingGraphPool->peek(phase);
        const auto t1 = Clock::now();
        totalPeekUs +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        ++peekCalls;
      } else {
        pendingGraphPool->peek(phase);
      }
    }
    if (timingEnabled && peekCalls > 0) {
      auto minPhase = *phasesToPeek.begin();
      auto maxPhase = *phasesToPeek.rbegin();
      std::cerr << "[PROTON] pendingGraphPool peek timing: phases=[" << minPhase
                << "," << maxPhase << "] peek_us=" << totalPeekUs
                << " peek_calls=" << peekCalls << std::endl;
    }
  }

  for (const auto &range : flushRanges) {
    auto *data = range.data;
    const size_t minPhaseToFlush = range.minPhaseToFlush;
    const size_t maxPhaseToFlush = range.maxPhaseToFlush;
    dataFlushedPhases[data] = maxPhaseToFlush;
    data->completePhase(maxPhaseToFlush);

    if (!periodicFlushEnabled)
      continue;

    PeriodicFlushStats stats{};
    periodicFlushDataPhases(*data, periodicFlushingFormat, minPhaseToFlush,
                            maxPhaseToFlush, timingEnabled, stats);
    periodicClearDataPhases(*data, maxPhaseToFlush, timingEnabled, stats);
    if (timingEnabled) {
      std::cerr << "[PROTON] periodicFlush timing: path=" << data->getPath()
                << " format=" << periodicFlushingFormat << " phases=["
                << minPhaseToFlush << "," << maxPhaseToFlush
                << "] toJsonString_us=" << stats.totalToJsonUs
                << " toJsonString_calls=" << stats.toJsonCalls
                << " toMsgPack_us=" << stats.totalToMsgPackUs
                << " toMsgPack_calls=" << stats.toMsgPackCalls
                << " json_write_us=" << stats.totalJsonWriteUs
                << " json_write_calls=" << stats.jsonWriteCalls
                << " msgpack_write_us=" << stats.totalMsgPackWriteUs
                << " msgpack_write_calls=" << stats.msgPackWriteCalls
                << " clear_us=" << stats.clearUs << std::endl;
    }
  }
}

} // namespace detail
} // namespace proton
