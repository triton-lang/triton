#include "Data/Data.h"
#include "Utility/String.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include <shared_mutex>

namespace proton {

void Data::initPhaseStore(PhaseStoreBase &store) {
  phaseStore = &store;
  currentPhasePtr = phaseStore->getOrCreatePtr(0);
  activePhases.insert(0);
}

size_t Data::advancePhase() {
  std::unique_lock<std::shared_mutex> lock(mutex);
  const auto nextPhase = currentPhase + 1;
  currentPhasePtr = phaseStore->getOrCreatePtr(nextPhase);
  activePhases.insert(nextPhase);
  currentPhase = nextPhase;
  return currentPhase;
}

void Data::clear(size_t phase, bool clearUpToPhase) {
  // No locking needed.
  // If phase == currentPhase, we expect users to call clear right after
  // deactivating the profiler, without any GPU events in between.
  // If phase < currentPhase, clearing a past phase is safe without locks.
  if (clearUpToPhase)
    phaseStore->clearUpToInclusive(phase);
  else
    phaseStore->clearPhase(phase);

  std::unique_lock<std::shared_mutex> lock(mutex);
  if (clearUpToPhase) {
    for (auto it = activePhases.begin(); it != activePhases.end();) {
      if (*it <= phase) {
        it = activePhases.erase(it);
      } else {
        ++it;
      }
    }
  } else {
    activePhases.erase(phase);
  }

  // In case the current phase is cleared, recreate its pointer.
  currentPhasePtr = phaseStore->getOrCreatePtr(currentPhase);
  activePhases.insert(currentPhase);
}

void Data::updateCompletePhase(size_t phase) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  if (completePhase == kNoCompletePhase || phase > completePhase)
    completePhase = phase;
}

bool Data::isPhaseComplete(size_t phase) const {
  std::shared_lock<std::shared_mutex> lock(mutex);
  return completePhase != kNoCompletePhase && completePhase >= phase;
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
      auto suffix = currentPhase == 0 ? "" : ".part_" + std::to_string(phase);
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
