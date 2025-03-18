#include "Profiler/Instrumentation/InstrumentationProfiler.h"

namespace proton {

void InstrumentationProfiler::doStart() {
  // Start the instrumentation profiler.
}

void InstrumentationProfiler::doFlush() {
  // Flush the instrumentation profiler.
}

void InstrumentationProfiler::doStop() {
  // Stop the instrumentation profiler.
}

void InstrumentationProfiler::initScopeIds(
    uint64_t functionId,
    const std::vector<std::pair<size_t, std::string>> &scopeIdPairs) {
  // Initialize the scope IDs.
}

void InstrumentationProfiler::enterInstrumentedOp(uint64_t functionId,
                                                  const uint8_t *buffer,
                                                  size_t size) {
  // Enter an instrumented operation.
}

void InstrumentationProfiler::exitInstrumentedOp(uint64_t functionId,
                                                 const uint8_t *buffer,
                                                 size_t size) {
  // Exit an instrumented operation.
}

} // namespace proton
