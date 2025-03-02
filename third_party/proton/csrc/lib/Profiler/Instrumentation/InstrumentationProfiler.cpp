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

InstrumentationProfiler::InstrumentationProfiler(const std::string &mode)
    : mode(mode) {}

InstrumentationProfiler::~InstrumentationProfiler() = default;

} // namespace proton
