#include "triton/Tools/LLVMOptions.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <condition_variable>
#include <map>
#include <mutex>
#include <new>

#ifndef _WIN32
#include <pthread.h>
#endif

using namespace mlir::triton::tools;

namespace {

using Settings = std::map<std::string, std::string>;

struct Registry {
  std::mutex mutex;
  std::condition_variable changed;

  // The override set currently applied to LLVM's registries. Empty whenever
  // `users` is zero.
  Settings active;
  // Live scopes sharing `active`.
  unsigned users = 0;
  // Scopes waiting to replace `active`. While any is pending, new scopes do not
  // join the current set, so a steady stream of compilations cannot starve one
  // that needs different options.
  unsigned switchesPending = 0;
  // Exclusive accesses waiting for the registry, and whether one holds it.
  unsigned exclusivesPending = 0;
  bool exclusiveHeld = false;
  // Incremented after fork so inherited guards do not mutate the child's
  // freshly initialized bookkeeping when their destructors run.
  std::uint64_t generation = 0;
};

Registry &registry();

#ifndef _WIN32
void lockRegistryBeforeFork() {
  Registry &reg = registry();
  std::unique_lock<std::mutex> lock(reg.mutex);
  // Block new scopes while waiting for an in-flight exclusive mutation (LLD)
  // to finish. Existing scopes only read the option registry, so they may be
  // snapshotted safely; their bookkeeping stays stable because the mutex
  // remains locked until fork completes.
  ++reg.exclusivesPending;
  reg.changed.wait(lock, [&] { return !reg.exclusiveHeld; });
  --reg.exclusivesPending;
  lock.release(); // No unlock--unlock done explicitly after forking.
}

void unlockRegistryInParent() {
  Registry &reg = registry();
  reg.mutex.unlock();
  reg.changed.notify_all();
}

// A forked child inherits the bookkeeping but not the threads that own it.
// The prepare handler locked the registry, so its containers are consistent.
void resetRegistryInChild() {
  Registry &reg = registry();
  llvm::cl::ResetAllOptionOccurrences();
  std::uint64_t generation = reg.generation + 1;
  new (&reg) Registry();
  reg.generation = generation;
}
#endif

Registry &registry() {
  // Never destroyed: scopes may outlive static destruction order.
  static Registry *instance = [] {
    auto *reg = new Registry();
#ifndef _WIN32
    pthread_atfork(lockRegistryBeforeFork, unlockRegistryInParent,
                   resetRegistryInChild);
#endif
    return reg;
  }();
  return *instance;
}

void applySettings(Registry &reg, Settings settings) {
  // Allocate before mutating any options so allocation failure cannot leave
  // the process-wide registry partially overridden.
  std::vector<std::string> arguments{"triton"};
  arguments.reserve(settings.size() + 1);
  for (const auto &[name, value] : settings)
    arguments.push_back("--" + name + "=" + value);
  std::vector<const char *> argv;
  argv.reserve(arguments.size());
  for (const std::string &argument : arguments)
    argv.push_back(argument.c_str());

  // getRegisteredOptions() only exposes cl::opt. The command-line parser also
  // handles TableGen library options, including codegen's start/stop passes,
  // and owns any argument strings retained by those options. Keep ignoring
  // unknown options, which may differ between LLVM revisions.
  llvm::cl::ParseCommandLineOptions(argv.size(), argv.data(), "",
                                    &llvm::nulls());
  reg.active = std::move(settings);
}

void restoreSettings(Registry &reg) {
  llvm::cl::ResetAllOptionOccurrences();
  reg.active.clear();
}

} // namespace

ScopedLLVMOptions::ScopedLLVMOptions(const std::vector<Setting> &settings) {
  Settings wanted;
  for (const Setting &setting : settings)
    wanted.insert_or_assign(setting.first, setting.second);

  Registry &reg = registry();
  std::unique_lock<std::mutex> lock(reg.mutex);
  bool pendingSwitch = false;
  reg.changed.wait(lock, [&] {
    if (reg.exclusiveHeld || reg.exclusivesPending)
      return false;
    if (reg.users == 0)
      return true;
    if (reg.active == wanted)
      return pendingSwitch || reg.switchesPending == 0;
    if (!pendingSwitch) {
      pendingSwitch = true;
      ++reg.switchesPending;
    }
    return false;
  });
  if (pendingSwitch)
    --reg.switchesPending;
  if (reg.users == 0)
    applySettings(reg, std::move(wanted));
  ++reg.users;
  registryGeneration = reg.generation;
}

ScopedLLVMOptions::~ScopedLLVMOptions() {
  Registry &reg = registry();
  std::lock_guard<std::mutex> lock(reg.mutex);
  if (registryGeneration != reg.generation)
    return;
  if (--reg.users == 0)
    restoreSettings(reg);
  reg.changed.notify_all();
}

ExclusiveLLVMOptionAccess::ExclusiveLLVMOptionAccess() {
  Registry &reg = registry();
  std::unique_lock<std::mutex> lock(reg.mutex);
  ++reg.exclusivesPending;
  reg.changed.wait(lock, [&] { return !reg.exclusiveHeld && reg.users == 0; });
  --reg.exclusivesPending;
  reg.exclusiveHeld = true;
  registryGeneration = reg.generation;
}

ExclusiveLLVMOptionAccess::~ExclusiveLLVMOptionAccess() {
  Registry &reg = registry();
  std::lock_guard<std::mutex> lock(reg.mutex);
  if (registryGeneration != reg.generation)
    return;
  llvm::cl::ResetAllOptionOccurrences();
  reg.exclusiveHeld = false;
  reg.changed.notify_all();
}
