#ifndef TRITON_TOOLS_DIAG_EMITTER_HPP
#define TRITON_TOOLS_DIAG_EMITTER_HPP
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <optional>

#define EMIT_PERF_WARNING(op, message)                                         \
  if (auto out = mlir::triton::DiagnosticEmitter::getPerfWarningStream(op)) {  \
    *out << message;                                                           \
  }

namespace mlir::triton {

class DiagnosticEmitter {
  // singleton pattern
private:
  inline static DiagnosticEmitter *instance{nullptr};
  bool shouldEmitPerfWarning;
  DiagnosticEmitter() : shouldEmitPerfWarning(false){};
  ~DiagnosticEmitter() = default;

public:
  DiagnosticEmitter(const DiagnosticEmitter &) = delete;
  DiagnosticEmitter &operator=(const DiagnosticEmitter &) = delete;

  static DiagnosticEmitter *getInstance() {
    if (!instance) {
      instance = new DiagnosticEmitter();
      if (tools::getBoolEnv("MLIR_ENABLE_REMARK")) {
        instance->shouldEmitPerfWarning = true;
      }
    }
    return instance;
  }

  static void setShouldEmitPerfWarning(bool shouldEmit) {
    DiagnosticEmitter::getInstance()->shouldEmitPerfWarning = shouldEmit;
  }

  static std::optional<InFlightDiagnostic>
  getPerfWarningStream(const OpState &op) {
    if (DiagnosticEmitter::getInstance()->shouldEmitPerfWarning) {
      return op->emitRemark();
    } else {
      return std::nullopt;
    }
  }
};
} // namespace mlir::triton
#endif
