#ifndef PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_
#define PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_

#include "Context/Context.h"
#include "Device.h"
#include "Metadata.h"
#include "Profiler/Profiler.h"
#include "Runtime.h"
#include "TraceDataIO/Parser.h"
#include "Utility/Singleton.h"

namespace proton {

class InstrumentationProfiler : public Profiler,
                                public InstrumentationInterface,
                                public OpInterface,
                                public Singleton<InstrumentationProfiler> {
public:
  InstrumentationProfiler() = default;
  virtual ~InstrumentationProfiler();

  InstrumentationProfiler *setMode(const std::vector<std::string> &mode);

protected:
  // Profiler
  virtual void doStart() override;
  virtual void doFlush() override;
  virtual void doStop() override;

  // InstrumentationInterface
  void initFunctionMetadata(
      uint64_t functionId, const std::string &functionName,
      const std::vector<std::pair<size_t, std::string>> &scopeIdNames,
      const std::vector<std::pair<size_t, size_t>> &scopeIdParentIds,
      const std::string &metadataPath) override;
  void enterInstrumentedOp(uint64_t streamId, uint64_t functionId,
                           uint8_t *buffer, size_t size) override;
  void exitInstrumentedOp(uint64_t streamId, uint64_t functionId,
                          uint8_t *buffer, size_t size) override;

  // OpInterface
  void startOp(const Scope &scope) override {
    for (auto data : getDataSet()) {
      auto scopeId = data->addOp(scope.scopeId, scope.name);
      dataScopeIdMap[data] = scopeId;
    }
  }
  void stopOp(const Scope &scope) override { dataScopeIdMap.clear(); }

private:
  std::shared_ptr<ParserConfig> getParserConfig(uint64_t functionId,
                                                size_t bufferSize) const;

  std::unique_ptr<Runtime> runtime;
  // device -> deviceStream
  std::map<void *, void *> deviceStreams;
  std::map<std::string, std::string> modeOptions;
  uint8_t *hostBuffer{nullptr};
  // functionId -> scopeId -> scopeName
  std::map<uint64_t, std::map<size_t, std::string>> functionScopeIdNames;
  // functionId -> scopeId -> contexts
  std::map<uint64_t, std::map<size_t, std::vector<Context>>>
      functionScopeIdContexts;
  ;
  // functionId -> functionName
  std::map<uint64_t, std::string> functionNames;
  // functionId -> metadata
  std::map<uint64_t, InstrumentationMetadata> functionMetadata;
  // data -> scopeId
  static thread_local std::map<Data *, size_t> dataScopeIdMap;
};

} // namespace proton

#endif // PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_
