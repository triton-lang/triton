#ifndef PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_
#define PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_

#include "Profiler/Profiler.h"

namespace proton {

class InstrumentationProfiler : public Profiler {
public:
  InstrumentationProfiler(const std::string &mode);
  virtual ~InstrumentationProfiler();

protected:
  virtual void doStart() override;
  virtual void doFlush() override;
  virtual void doStop() override;

private:
  const std::string mode;
};

} // namespace proton

#endif // PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_
