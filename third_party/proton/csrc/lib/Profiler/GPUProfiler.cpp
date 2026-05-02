#include "Profiler/GPUProfiler.h"
#include "Profiler/Graph.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

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

    size_t minPhaseToFlush = 0;
    if (flushedPhaseIt == dataFlushedPhases.end() ||
        flushedPhaseIt->second == Data::kNoCompletePhase) {
      minPhaseToFlush = 0;
    } else {
      const auto flushedPhase = flushedPhaseIt->second;
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

template <typename T>
void timingWrapper(const bool timingEnabled, uint64_t &totalUs, size_t &calls,
                   const T &func) {
  if (timingEnabled) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    func();
    const auto t1 = Clock::now();
    totalUs +=
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    ++calls;
  } else {
    func();
  }
}

bool isProcFdPath(const std::string &path) {
  return path.rfind("/proc/self/fd/", 0) == 0;
}

int parseProcFd(const std::string &path) {
  if (!isProcFdPath(path))
    return -1;
  return std::stoi(path.substr(std::string("/proc/self/fd/").size()));
}

void writeToFd(int fd, const uint8_t *data, size_t size) {
  size_t totalWritten = 0;
  while (totalWritten < size) {
    const auto written = ::write(fd, data + totalWritten, size - totalWritten);
    if (written < 0) {
      if (errno == EINTR)
        continue;
      throw std::system_error(errno, std::generic_category(),
                              "[PROTON] Failed to write periodic profile data");
    }
    totalWritten += static_cast<size_t>(written);
  }
}

void writeToFd(int fd, const std::string &data) {
  writeToFd(fd, reinterpret_cast<const uint8_t *>(data.data()), data.size());
}

void periodicFlushDataPhases(Data &data,
                             const std::string &periodicFlushingFormat,
                             size_t minPhaseToFlush, size_t maxPhaseToFlush,
                             const bool timingEnabled,
                             PeriodicFlushStats &stats) {
  const auto &path = data.getPath();
  const bool streamToProcFd = isProcFdPath(path);
  const int outputFd = parseProcFd(path);

  for (auto startPhase = minPhaseToFlush; startPhase <= maxPhaseToFlush;
       startPhase++) {
    auto pathWithPhase = streamToProcFd
                             ? path
                             : path + ".part_" + std::to_string(startPhase) +
                                   "." + periodicFlushingFormat;

    if (periodicFlushingFormat == "hatchet" ||
        periodicFlushingFormat == "chrome_trace") {
      std::string jsonStr;
      timingWrapper(timingEnabled, stats.totalToJsonUs, stats.toJsonCalls,
                    [&]() { jsonStr = data.toJsonString(startPhase); });

      if (streamToProcFd) {
        timingWrapper(timingEnabled, stats.totalJsonWriteUs,
                      stats.jsonWriteCalls,
                      [&]() { writeToFd(outputFd, jsonStr); });
      } else {
        timingWrapper(
            timingEnabled, stats.totalJsonWriteUs, stats.jsonWriteCalls, [&]() {
              std::ofstream ofs(pathWithPhase, std::ios::out | std::ios::trunc);
              ofs << jsonStr;
              ofs.flush();
            });
      }
    } else if (periodicFlushingFormat == "hatchet_msgpack") {
      std::vector<uint8_t> msgPack;
      timingWrapper(timingEnabled, stats.totalToMsgPackUs, stats.toMsgPackCalls,
                    [&]() { msgPack = data.toMsgPack(startPhase); });

      if (streamToProcFd) {
        timingWrapper(
            timingEnabled, stats.totalMsgPackWriteUs, stats.msgPackWriteCalls,
            [&]() { writeToFd(outputFd, msgPack.data(), msgPack.size()); });
      } else {
        timingWrapper(timingEnabled, stats.totalMsgPackWriteUs,
                      stats.msgPackWriteCalls, [&]() {
                        std::ofstream ofs(pathWithPhase, std::ios::out |
                                                             std::ios::binary |
                                                             std::ios::trunc);
                        ofs.write(
                            reinterpret_cast<const char *>(msgPack.data()),
                            msgPack.size());
                      });
      }
    }
  }
}

void periodicClearDataPhases(Data &data, size_t maxPhaseToFlush,
                             const bool timingEnabled,
                             PeriodicFlushStats &stats) {
  size_t clearCalls = 0;
  timingWrapper(timingEnabled, stats.clearUs, clearCalls, [&]() {
    data.clear(maxPhaseToFlush, /*clearUpToPhase=*/true);
  });
}

} // namespace

void setPeriodicFlushingMode(bool &periodicFlushingEnabled,
                             std::string &periodicFlushingFormat,
                             const std::vector<std::string> &modeAndOptions,
                             const char *profilerName) {
  periodicFlushingEnabled = true;
  if (modeAndOptions.size() < 2) {
    periodicFlushingFormat = "hatchet";
    return;
  }

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
        &dataPhases,
    PendingGraphPool *pendingGraphPool) {
  static const bool timingEnabled =
      getBoolEnv("PROTON_DATA_FLUSH_TIMING", false);
  auto [flushRanges, phasesToPeek] = computeFlushRangesAndPeekPhases(
      dataFlushedPhases, dataPhases, pendingGraphPool != nullptr);
  if (pendingGraphPool) {
    uint64_t totalPeekUs = 0;
    size_t peekCalls = 0;
    for (const auto phase : phasesToPeek) {
      timingWrapper(timingEnabled, totalPeekUs, peekCalls,
                    [&]() { pendingGraphPool->peek(phase); });
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
