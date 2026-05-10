#ifndef PROTON_PROFILER_INSTRUMENTATION_METADATA_H_
#define PROTON_PROFILER_INSTRUMENTATION_METADATA_H_

#include <vector>

namespace proton {

class InstrumentationMetadata {

public:
  InstrumentationMetadata(const std::string &metadataPath)
      : metadataPath(metadataPath) {
    parse();
  }

  size_t getScratchMemorySize() const { return scratchMemorySize; }

  size_t getNumWarps() const { return numWarps; }

private:
  void parse();

  const std::string metadataPath;
  size_t scratchMemorySize{};
  size_t numWarps{};
};

} // namespace proton

#endif // PROTON_PROFILER_INSTRUMENTATION_METADATA_H_
