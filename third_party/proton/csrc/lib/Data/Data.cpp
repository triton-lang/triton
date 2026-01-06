#include "Data/Data.h"
#include "Utility/String.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include <shared_mutex>

namespace proton {

size_t Data::advancePhase() {
  std::unique_lock<std::shared_mutex> lock(mutex);
  doAdvancePhase();
  auto nextPhase = currentPhase + 1;
  activePhases.insert(nextPhase);
  return ++currentPhase;
}

void Data::clear(size_t phase) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  doClear(phase);
  activePhases.clear();
  for (phase += 1; phase <= currentPhase; phase++)
    activePhases.insert(phase);
}

std::string Data::toJsonString(size_t phase) const {
  std::shared_lock<std::shared_mutex> lock(mutex);
  return doToJsonString(phase);
}

std::vector<uint8_t> Data::toMsgPack(size_t phase) const {
  std::shared_lock<std::shared_mutex> lock(mutex);
  return doToMsgPack(phase);
}

void Data::dump(const std::string &outputFormat) {
  std::shared_lock<std::shared_mutex> lock(mutex);

  OutputFormat outputFormatEnum = outputFormat.empty()
                                      ? getDefaultOutputFormat()
                                      : parseOutputFormat(outputFormat);

  for (auto phase : activePhases) {
    std::unique_ptr<std::ostream> out;
    if (path.empty() || path == "-") {
      out.reset(new std::ostream(std::cout.rdbuf())); // Redirecting to cout
    } else {
      auto suffix =
          getCurrentPhase() == 0 ? "" : ".part_" + std::to_string(phase);
      const auto filePath =
          path + suffix + "." + outputFormatToString(outputFormatEnum);
      const auto fileMode =
          (outputFormatEnum == OutputFormat::HatchetMsgPack)
              ? (std::ios::out | std::ios::binary | std::ios::trunc)
              : (std::ios::out | std::ios::trunc);
      out.reset(
          new std::ofstream(filePath, fileMode)); // Opening a file for output
    }

    doDump(*out, outputFormatEnum, phase);
  }
}

OutputFormat parseOutputFormat(const std::string &outputFormat) {
  if (toLower(outputFormat) == "hatchet") {
    return OutputFormat::Hatchet;
  } else if (toLower(outputFormat) == "hatchet_msgpack") {
    return OutputFormat::HatchetMsgPack;
  } else if (toLower(outputFormat) == "chrome_trace") {
    return OutputFormat::ChromeTrace;
  } else {
    throw std::runtime_error("Unknown output format: " + outputFormat);
  }
}

const std::string outputFormatToString(OutputFormat outputFormat) {
  if (outputFormat == OutputFormat::Hatchet) {
    return "hatchet";
  } else if (outputFormat == OutputFormat::HatchetMsgPack) {
    return "hatchet_msgpack";
  } else if (outputFormat == OutputFormat::ChromeTrace) {
    return "chrome_trace";
  }
  throw std::runtime_error("Unknown output format: " +
                           std::to_string(static_cast<int>(outputFormat)));
}

} // namespace proton
