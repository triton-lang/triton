#include "plugin_registration.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Signals.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace py = nanobind;

#define FOR_EACH_1(MACRO, X) MACRO(X)
#define FOR_EACH_2(MACRO, X, ...) MACRO(X) FOR_EACH_1(MACRO, __VA_ARGS__)
#define FOR_EACH_3(MACRO, X, ...) MACRO(X) FOR_EACH_2(MACRO, __VA_ARGS__)
#define FOR_EACH_4(MACRO, X, ...) MACRO(X) FOR_EACH_3(MACRO, __VA_ARGS__)
#define FOR_EACH_5(MACRO, X, ...) MACRO(X) FOR_EACH_4(MACRO, __VA_ARGS__)

#define FOR_EACH_NARG(...) FOR_EACH_NARG_(__VA_ARGS__, FOR_EACH_RSEQ_N())
#define FOR_EACH_NARG_(...) FOR_EACH_ARG_N(__VA_ARGS__)
#define FOR_EACH_ARG_N(_1, _2, _3, _4, _5, N, ...) N
#define FOR_EACH_RSEQ_N() 5, 4, 3, 2, 1, 0

#define CONCATENATE(x, y) CONCATENATE1(x, y)
#define CONCATENATE1(x, y) x##y

#define FOR_EACH(MACRO, ...)                                                   \
  CONCATENATE(FOR_EACH_, FOR_EACH_NARG_HELPER(__VA_ARGS__))(MACRO, __VA_ARGS__)
#define FOR_EACH_NARG_HELPER(...) FOR_EACH_NARG(__VA_ARGS__)

// New macro to remove parentheses
#define REMOVE_PARENS(...) __VA_ARGS__

// Intermediate macro to ensure correct expansion
#define FOR_EACH_P_INTERMEDIATE(MACRO, ...) FOR_EACH(MACRO, __VA_ARGS__)

// Modified FOR_EACH to handle parentheses
#define FOR_EACH_P(MACRO, ARGS_WITH_PARENS)                                    \
  FOR_EACH_P_INTERMEDIATE(MACRO, REMOVE_PARENS ARGS_WITH_PARENS)

#define DECLARE_BACKEND(name) void init_triton_##name(nanobind::module_ &m);

#define INIT_BACKEND(name)                                                     \
  {                                                                            \
    auto sub = m.def_submodule(#name);                                         \
    init_triton_##name(sub);                                                   \
  }

void init_triton_env_vars(nanobind::module_ &m);
void init_triton_ir(nanobind::module_ &m);
void init_triton_llvm(nanobind::module_ &m);
void init_triton_interpreter(nanobind::module_ &m);
void init_triton_passes(nanobind::module_ &m);
void init_triton_stacktrace_hook(nanobind::module_ &m);
void init_gluon_ir(nanobind::module_ &m);
void init_gsan_testing(nanobind::module_ &m);
void init_linear_layout(nanobind::module_ &m);
void init_native_specialize(nanobind::module_ &m);
FOR_EACH_P(DECLARE_BACKEND, TRITON_BACKENDS_TUPLE)

NB_MODULE(libtriton, m) {
  m.doc() = "Python bindings to the C++ Triton API";
  init_triton_stacktrace_hook(m);
  init_triton_env_vars(m);
  init_native_specialize(m);
  auto ir_m = m.def_submodule("ir");
  init_triton_ir(ir_m);
  auto passes_m = m.def_submodule("passes");
  init_triton_passes(passes_m);
  auto interpreter_m = m.def_submodule("interpreter");
  init_triton_interpreter(interpreter_m);
  auto llvm_m = m.def_submodule("llvm");
  init_triton_llvm(llvm_m);
  auto gsan_m = m.def_submodule("gsan_testing");
  init_gsan_testing(gsan_m);
  auto ll_m = m.def_submodule("linear_layout");
  init_linear_layout(ll_m);
  auto gluon_m = m.def_submodule("gluon_ir");
  init_gluon_ir(gluon_m);
  FOR_EACH_P(INIT_BACKEND, TRITON_BACKENDS_TUPLE)

  // Single, top-level entry point for extending Triton with a plugin. This must
  // be defined after the module tree (`ir`, `passes`, ...) has been built so
  // that it can append the plugin's dialects, operations, and passes in each
  // place. The top-level module is captured by value (its lifetime is owned by
  // nanobind, which cleans it up while the interpreter is still valid) and
  // forwarded so that `extendTritonWith` can walk the module tree at call time.
  // See `extendTritonWith` in `ir.cc`.
  m.def(
      "extend_with",
      [m](const std::string &path) { extendTritonWith(m, path); },
      "Given a path to a Triton extension, load it once and register its "
      "dialects, custom operations, and passes.");
}
