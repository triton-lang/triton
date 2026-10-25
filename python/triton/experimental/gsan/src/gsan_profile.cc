#include "AllocationProfile.h"

#include <nanobind/nanobind.h>

void init_gsan_profile(nanobind::module_ &m) {
  m.doc() = "GSan allocation profiling metadata layout";
  m.attr("REPEATED_WRITE") = gsan_profile::kRepeatedWrite;
  m.attr("READ_BEFORE_WRITE") = gsan_profile::kReadBeforeWrite;
  m.attr("ATOMIC") = gsan_profile::kAtomic;
  m.attr("READ") = gsan_profile::kRead;
  m.attr("WRITE") = gsan_profile::kWrite;
  m.attr("WRITTEN_MASK") = gsan_profile::kWrittenMask;
  m.attr("FLAGS_MASK") = gsan_profile::kFlagsMask;
  m.attr("GRANULARITY_SHIFT") = gsan_profile::kGranularityShift;
  m.attr("NUM_GRANULARITIES") = gsan_profile::kNumGranularities;
  m.attr("GRANULARITY") = gsan_profile::kGranularity;
  m.attr("CELL_SIZE_BYTES") = sizeof(gsan_profile::ProfileCell);
}
