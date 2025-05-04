#include <fstream>

#include "Profiler/Instrumentation/Metadata.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace proton {

void InstrumentationMetadata::parse() {
  std::ifstream metadataFile(metadataPath);
  if (!metadataFile.is_open()) {
    throw std::runtime_error("Failed to open metadata file: " + metadataPath);
  }

  json metadataJson;
  metadataFile >> metadataJson;

  if (metadataJson.contains("profile_scratch_size")) {
    scratchMemorySize = metadataJson["profile_scratch_size"].get<size_t>();
  }

  // FIXME: this is wrong
  if (metadataJson.contains("num_warps")) {
    numWarps = metadataJson["num_warps"].get<size_t>();
  }
}

} // namespace proton
