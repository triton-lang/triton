#include "Profiler/GPUProfiler.h"
#include "Data/TreeData.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace proton {
namespace detail {

std::vector<std::string> splitFormats(const std::string &format) {
  std::vector<std::string> formats;
  std::string token;
  token.reserve(format.size());
  for (char ch : format) {
    if (ch == '+' || ch == ',') {
      if (!token.empty()) {
        formats.push_back(token);
        token.clear();
      }
      continue;
    }
    token.push_back(ch);
  }
  if (!token.empty()) {
    formats.push_back(token);
  }
  if (formats.empty()) {
    formats.push_back(format);
  }
  return formats;
}

struct FormatSpec {
  std::string name;
  std::optional<PeriodicFlushTarget> target;
};

std::vector<FormatSpec>
parseFormatSpecs(const std::string &format, PeriodicFlushTarget defaultTarget) {
  std::vector<FormatSpec> specs;
  for (const auto &token : splitFormats(format)) {
    auto atPos = token.find('@');
    if (atPos == std::string::npos) {
      specs.push_back({token, defaultTarget});
      continue;
    }
    auto name = token.substr(0, atPos);
    auto targetStr = token.substr(atPos + 1);
    if (targetStr == "disk") {
      specs.push_back({name, PeriodicFlushTarget::Disk});
    } else if (targetStr == "buffer") {
      specs.push_back({name, PeriodicFlushTarget::Buffer});
    } else {
      throw std::invalid_argument(std::string("[PROTON] ") +
                                  "unsupported target: " + targetStr);
    }
  }
  return specs;
}

void setPeriodicFlushingMode(bool &periodicFlushingEnabled,
                             std::string &periodicFlushingFormat,
                             PeriodicFlushTarget &periodicFlushingTarget,
                             const std::vector<std::string> &modeAndOptions,
                             const char *profilerName) {
  periodicFlushingEnabled = true;
  periodicFlushingFormat = "hatchet";
  periodicFlushingTarget = PeriodicFlushTarget::Disk;

  const auto validateFormatSpecs = [&](const std::string &format,
                                       PeriodicFlushTarget defaultTarget) {
    const auto specs = parseFormatSpecs(format, defaultTarget);
    std::set<std::string> seenFormats;
    for (const auto &spec : specs) {
      const auto &fmt = spec.name;
      if (fmt != "hatchet_msgpack" && fmt != "chrome_trace" &&
          fmt != "hatchet" && fmt != "path_metrics") {
        throw std::invalid_argument(std::string("[PROTON] ") + profilerName +
                                    ": unsupported format: " + fmt);
      }
      if (!seenFormats.emplace(fmt).second) {
        throw std::invalid_argument(std::string("[PROTON] ") + profilerName +
                                    ": duplicate format: " + fmt);
      }
      if (spec.target == PeriodicFlushTarget::Buffer && fmt != "path_metrics") {
        throw std::invalid_argument(std::string("[PROTON] ") + profilerName +
                                    ": target=buffer only supports path_metrics");
      }
      if (fmt == "path_metrics" && spec.target == PeriodicFlushTarget::Disk) {
        throw std::invalid_argument(std::string("[PROTON] ") + profilerName +
                                    ": format=path_metrics requires target=buffer");
      }
    }
  };

  for (size_t i = 1; i < modeAndOptions.size(); ++i) {
    auto delimiterPos = modeAndOptions[i].find('=');
    if (delimiterPos == std::string::npos) {
      throw std::invalid_argument(std::string("[PROTON] ") + profilerName +
                                  ": unsupported option: " + modeAndOptions[i]);
    }
    const std::string key = modeAndOptions[i].substr(0, delimiterPos);
    const std::string value = modeAndOptions[i].substr(delimiterPos + 1);
    if (key == "format") {
      periodicFlushingFormat = value;
    } else if (key == "path_metrics_rules") {
      TreeData::setPathMetricsRules(value);
    } else if (key == "target") {
      if (value == "disk") {
        periodicFlushingTarget = PeriodicFlushTarget::Disk;
      } else if (value == "buffer") {
        periodicFlushingTarget = PeriodicFlushTarget::Buffer;
      } else {
        throw std::invalid_argument(std::string("[PROTON] ") + profilerName +
                                    ": unsupported target: " + value);
      }
    } else {
      throw std::invalid_argument(std::string("[PROTON] ") + profilerName +
                                  ": unsupported option key: " + key);
    }
  }
  validateFormatSpecs(periodicFlushingFormat, periodicFlushingTarget);
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
    const PeriodicFlushTarget periodicFlushingTarget,
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

    const auto specs =
        parseFormatSpecs(periodicFlushingFormat, periodicFlushingTarget);
    std::optional<PeriodicFlushTarget> hatchetTarget;
    std::optional<PeriodicFlushTarget> chromeTraceTarget;
    std::optional<PeriodicFlushTarget> msgPackTarget;
    std::optional<PeriodicFlushTarget> pathMetricsTarget;
    for (const auto &spec : specs) {
      if (spec.name == "hatchet") {
        hatchetTarget = spec.target;
      } else if (spec.name == "chrome_trace") {
        chromeTraceTarget = spec.target;
      } else if (spec.name == "hatchet_msgpack") {
        msgPackTarget = spec.target;
      } else if (spec.name == "path_metrics") {
        pathMetricsTarget = spec.target;
      }
    }

    for (auto startPhase = minPhaseToFlush; startPhase <= maxPhaseToFlush;
         startPhase++) {
      if (pathMetricsTarget.has_value()) {
        auto metrics = data->toPathMetrics(startPhase);
        if (!metrics.empty()) {
          data->enqueueFlushedPathMetrics(startPhase, std::move(metrics));
        }
      }

      if (hatchetTarget.has_value() || chromeTraceTarget.has_value()) {
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

        const auto writeJson = [&](const std::string &formatSuffix) {
          auto pathWithPhase =
              path + ".part_" + std::to_string(startPhase) + "." + formatSuffix;
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
        };
        if (hatchetTarget == PeriodicFlushTarget::Disk) {
          writeJson("hatchet");
        }
        if (chromeTraceTarget == PeriodicFlushTarget::Disk) {
          writeJson("chrome_trace");
        }
      }

      if (msgPackTarget.has_value()) {
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
          auto pathWithPhase = path + ".part_" +
                               std::to_string(startPhase) +
                               ".hatchet_msgpack";
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
          auto pathWithPhase = path + ".part_" +
                               std::to_string(startPhase) +
                               ".hatchet_msgpack";
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
