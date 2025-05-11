#ifndef PROTON_COMMON_TRACE_WRITER_H_
#define PROTON_COMMON_TRACE_WRITER_H_

#include "CircularLayoutParser.h"
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace proton {

// TODO(fywkevin): this time gap to offset multiple kernels is not needed after
// we have the global time.
const uint32_t kKernelTimeGap = 10000000;

struct KernelMetadata {
  std::map<int, std::string> scopeName;
  std::string kernelName;
};

using KernelTrace = std::pair<CircularLayoutParserResult *, KernelMetadata *>;

// StreamTraceWriter handles trace dumping for a single cuda stream.
// If we have multiple stream, simply having a for loop to write to multiple
// files (one for each stream). Other types of per-stream trace writers could
// subclass the StreamTraceWriter such as StreamPerfettoTraceWriter that
// produces a protobuf format trace.
class StreamTraceWriter {
public:
  explicit StreamTraceWriter(const std::vector<KernelTrace> &streamTrace,
                             const std::string &path);

  virtual ~StreamTraceWriter() = default;

  void dump();

protected:
  virtual void write(std::ofstream &outfile) = 0;

  const std::string path;
  const std::vector<KernelTrace> &streamTrace;
};

class StreamChromeTraceWriter : public StreamTraceWriter {
public:
  explicit StreamChromeTraceWriter(const std::vector<KernelTrace> &streamTrace,
                                   const std::string &path);

private:
  void write(std::ofstream &outfile) override final;
  void writeKernel(std::stringstream &outstream, const KernelTrace &kernelTrace,
                   uint32_t kernelTimeStart);

  const std::vector<std::string> kChromeColor = {"cq_build_passed",
                                                 "cq_build_failed",
                                                 "thread_state_iowait",
                                                 "thread_state_running",
                                                 "thread_state_runnable",
                                                 "thread_state_unknown",
                                                 "rail_response",
                                                 "rail_idle",
                                                 "rail_load",
                                                 "cq_build_attempt_passed",
                                                 "cq_build_attempt_failed"};
};

} // namespace proton

#endif // PROTON_COMMON_TRACE_WRITER_H_
