#ifndef PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_
#define PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_

#include "Context/Context.h"
#include "Profiler/Profiler.h"
#include "Utility/Singleton.h"

namespace proton {

class InstrumentationProfiler : public Profiler,
                                public InstrumentationInterface,
                                public Singleton<InstrumentationProfiler> {
public:
  InstrumentationProfiler() = default;
  virtual ~InstrumentationProfiler() = default;

  InstrumentationProfiler *setMode(const std::string &mode) {
    this->mode = mode;
    return this;
  }

protected:
  // Profiler
  virtual void doStart() override;
  virtual void doFlush() override;
  virtual void doStop() override;

  // InstrumentationInterface
  void initScopeIds(
      uint64_t functionId,
      const std::vector<std::pair<size_t, std::string>> &scopeIdPairs) override;
  void enterInstrumentedOp(uint64_t functionId, const uint8_t *buffer,
                           size_t size) override;
  void exitInstrumentedOp(uint64_t functionId, const uint8_t *buffer,
                          size_t size) override;

private:
  std::string mode;
};

} // namespace proton

#endif // PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_
