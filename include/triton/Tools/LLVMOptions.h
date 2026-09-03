#ifndef TRITON_TOOLS_LLVMOPTIONS_H
#define TRITON_TOOLS_LLVMOPTIONS_H

#include <string>
#include <utility>
#include <vector>

namespace mlir::triton::tools {

// LLVM command line options (cl::opt) are process-wide globals with no
// internal locking. Codegen reads them at unpredictable points:
// TargetPassConfig snapshots the start/stop passes when it is built, and the
// AMDGPU schedulers query occurrence counts once per function. LLD goes further
// and rewrites every registered option through cl::ResetAllOptionOccurrences()
// on each link. The two classes below are the only sanctioned way for Triton to
// touch that registry.
//
// A ScopedLLVMOptions holds a set of option overrides for its lifetime. The
// registry carries exactly one such set at a time: scopes requesting an
// identical set share it and run in parallel, while a scope requesting a
// different set (including the empty set, which asks for the built-in
// defaults) waits until the previous set has no users left. Every LLVM pass
// pipeline run must happen inside a scope, even one without overrides, so that
// the registry is never rewritten underneath it.
//
// A thread must not request an ExclusiveLLVMOptionAccess while it holds a
// scope; the exclusive access would wait for that scope forever.
class ScopedLLVMOptions {
public:
  // An option name and its value, spelled as on the command line.
  using Setting = std::pair<std::string, std::string>;

  // Later settings for the same option win, like repeated command line flags.
  // Options unknown to LLVM are ignored.
  explicit ScopedLLVMOptions(const std::vector<Setting> &settings);
  ~ScopedLLVMOptions();

  ScopedLLVMOptions(const ScopedLLVMOptions &) = delete;
  ScopedLLVMOptions &operator=(const ScopedLLVMOptions &) = delete;
};

// Exclusive access to the option registry for code that rewrites it wholesale,
// such as LLD. Construction waits until no ScopedLLVMOptions is alive; scopes
// requested in the meantime wait until the exclusive access is released.
class ExclusiveLLVMOptionAccess {
public:
  ExclusiveLLVMOptionAccess();
  ~ExclusiveLLVMOptionAccess();

  ExclusiveLLVMOptionAccess(const ExclusiveLLVMOptionAccess &) = delete;
  ExclusiveLLVMOptionAccess &
  operator=(const ExclusiveLLVMOptionAccess &) = delete;
};

} // namespace mlir::triton::tools

#endif // TRITON_TOOLS_LLVMOPTIONS_H
