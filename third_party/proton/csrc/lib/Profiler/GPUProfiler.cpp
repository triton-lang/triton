#include "Profiler/GPUProfiler.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace proton {
namespace detail {

void setPeriodicFlushingMode(bool &periodicFlushingEnabled,
                             std::string &periodicFlushingFormat,
                             const std::vector<std::string> &modeAndOptions,
                             const char *profilerName) {
  periodicFlushingEnabled = true;
  if (modeAndOptions.size() < 2)
    periodicFlushingFormat = "hatchet";

  auto delimiterPos = modeAndOptions[1].find('=');
  if (delimiterPos != std::string::npos) {
    const std::string key = modeAndOptions[1].substr(0, delimiterPos);
    const std::string value = modeAndOptions[1].substr(delimiterPos + 1);
    if (key != "format") {
      throw std::invalid_argument(std::string("[PROTON] ") + profilerName +
                                  ": unsupported option key: " + key);
    }
    if (value != "hatchet_msgpack" && value != "chrome_trace" &&
        value != "hatchet") {
      throw std::invalid_argument(std::string("[PROTON] ") + profilerName +
                                  ": unsupported format: " + value);
    }
    periodicFlushingFormat = value;
  } else {
    periodicFlushingFormat = "hatchet";
  }
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
        &dataPhases) {
  static const bool timingEnabled =
      getBoolEnv("PROTON_PERIODIC_FLUSH_TIMING", false);
  using Clock = std::chrono::steady_clock;

  for (auto [data, phase] : dataPhases) {
    if (phase.second == 0)
      continue;

    size_t minPhaseToFlush = 0;
    size_t maxPhaseToFlush = 0;
    auto flushedPhaseIt = dataFlushedPhases.find(data);
    if (flushedPhaseIt == dataFlushedPhases.end() ||
        flushedPhaseIt->second == Data::kNoFlushedPhase) {
      minPhaseToFlush = 0;
      maxPhaseToFlush = phase.second - 1;
    } else {
      auto flushedPhase = flushedPhaseIt->second;
      if (phase.second - 1 <= flushedPhase)
        continue;
      minPhaseToFlush = flushedPhase + 1;
      maxPhaseToFlush = phase.second - 1;
    }
    // dataFlushedPhases should be recorded also here to avoid profiler's data
    // unregistered
    dataFlushedPhases[data] = maxPhaseToFlush;
    data->updateFlushedPhase(maxPhaseToFlush);
    if (!periodicFlushEnabled)
      continue;

    auto &path = data->getPath();
    uint64_t totalToJsonUs = 0;
    uint64_t totalToMsgPackUs = 0;
    uint64_t totalJsonWriteUs = 0;
    uint64_t totalMsgPackWriteUs = 0;
    size_t toJsonCalls = 0;
    size_t toMsgPackCalls = 0;
    size_t jsonWriteCalls = 0;
    size_t msgPackWriteCalls = 0;

    for (auto startPhase = minPhaseToFlush; startPhase <= maxPhaseToFlush;
         startPhase++) {
      auto pathWithPhase = path + ".part_" + std::to_string(startPhase) + "." +
                           periodicFlushingFormat;

      if (periodicFlushingFormat == "hatchet" ||
          periodicFlushingFormat == "chrome_trace") {
        std::string jsonStr;
        if (timingEnabled) {
          const auto t0 = Clock::now();
          jsonStr = data->toJsonString(startPhase);
          const auto t1 = Clock::now();
          totalToJsonUs +=
              std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count();
          ++toJsonCalls;
        } else {
          jsonStr = data->toJsonString(startPhase);
        }

        if (timingEnabled) {
          const auto t0 = Clock::now();
          std::ofstream ofs(pathWithPhase, std::ios::out | std::ios::trunc);
          ofs << jsonStr;
          ofs.flush();
          const auto t1 = Clock::now();
          totalJsonWriteUs +=
              std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count();
          ++jsonWriteCalls;
        } else {
          std::ofstream ofs(pathWithPhase, std::ios::out | std::ios::trunc);
          ofs << jsonStr;
        }
      } else if (periodicFlushingFormat == "hatchet_msgpack") {
        std::vector<uint8_t> msgPack;
        if (timingEnabled) {
          const auto t0 = Clock::now();
          msgPack = data->toMsgPack(startPhase);
          const auto t1 = Clock::now();
          totalToMsgPackUs +=
              std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count();
          ++toMsgPackCalls;
        } else {
          msgPack = data->toMsgPack(startPhase);
        }

        if (timingEnabled) {
          const auto t0 = Clock::now();
          std::ofstream ofs(pathWithPhase,
                            std::ios::out | std::ios::binary | std::ios::trunc);
          ofs.write(reinterpret_cast<const char *>(msgPack.data()),
                    msgPack.size());
          ofs.flush();
          const auto t1 = Clock::now();
          totalMsgPackWriteUs +=
              std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count();
          ++msgPackWriteCalls;
        } else {
          std::ofstream ofs(pathWithPhase,
                            std::ios::out | std::ios::binary | std::ios::trunc);
          ofs.write(reinterpret_cast<const char *>(msgPack.data()),
                    msgPack.size());
        }
      }
    }

    uint64_t clearUs = 0;
    if (timingEnabled) {
      const auto t0 = Clock::now();
      data->clear(maxPhaseToFlush);
      const auto t1 = Clock::now();
      clearUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                    .count();
      std::cerr << "[PROTON] periodicFlush timing: path=" << path
                << " format=" << periodicFlushingFormat << " phases=["
                << minPhaseToFlush << "," << maxPhaseToFlush
                << "] toJsonString_us=" << totalToJsonUs
                << " toJsonString_calls=" << toJsonCalls
                << " toMsgPack_us=" << totalToMsgPackUs
                << " toMsgPack_calls=" << toMsgPackCalls
                << " json_write_us=" << totalJsonWriteUs
                << " json_write_calls=" << jsonWriteCalls
                << " msgpack_write_us=" << totalMsgPackWriteUs
                << " msgpack_write_calls=" << msgPackWriteCalls
                << " clear_us=" << clearUs << std::endl;
    } else {
      data->clear(maxPhaseToFlush);
    }
  }
}

} // namespace detail
} // namespace proton
