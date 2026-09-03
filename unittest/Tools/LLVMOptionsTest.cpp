#include "triton/Tools/LLVMOptions.h"

#include "llvm/Support/CommandLine.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

using mlir::triton::tools::ExclusiveLLVMOptionAccess;
using mlir::triton::tools::ScopedLLVMOptions;

namespace {

// Registered into the same process-wide registry as LLVM's own options.
llvm::cl::opt<bool> boolOption("triton-test-bool-option", llvm::cl::init(false),
                               llvm::cl::Hidden);
llvm::cl::opt<std::string> stringOption("triton-test-string-option",
                                        llvm::cl::init("default"),
                                        llvm::cl::Hidden);

using Settings = std::vector<ScopedLLVMOptions::Setting>;

// Runs `body` on another thread so tests can check that it blocks.
class BackgroundTask {
public:
  explicit BackgroundTask(std::function<void()> body)
      : thread([this, body = std::move(body)] {
          body();
          finished = true;
        }) {}
  ~BackgroundTask() { join(); }

  // Long enough for a thread that is not blocked to make progress.
  bool finishedWithinGracePeriod() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return finished;
  }

  void join() {
    if (thread.joinable())
      thread.join();
  }

private:
  std::atomic<bool> finished{false};
  std::thread thread;
};

TEST(LLVMOptions, ScopeAppliesAndRestores) {
  {
    ScopedLLVMOptions scope(
        Settings{{"triton-test-bool-option", "true"},
                 {"triton-test-string-option", "override"}});
    EXPECT_TRUE(boolOption.getValue());
    EXPECT_EQ(boolOption.getNumOccurrences(), 1);
    EXPECT_EQ(stringOption.getValue(), "override");
    EXPECT_EQ(stringOption.getNumOccurrences(), 1);
  }
  EXPECT_FALSE(boolOption.getValue());
  EXPECT_EQ(boolOption.getNumOccurrences(), 0);
  EXPECT_EQ(stringOption.getValue(), "default");
  EXPECT_EQ(stringOption.getNumOccurrences(), 0);
}

TEST(LLVMOptions, LastSettingWins) {
  ScopedLLVMOptions scope(Settings{{"triton-test-string-option", "first"},
                                   {"triton-test-string-option", "second"}});
  EXPECT_EQ(stringOption.getValue(), "second");
  EXPECT_EQ(stringOption.getNumOccurrences(), 1);
}

TEST(LLVMOptions, UnknownOptionsAreIgnored) {
  ScopedLLVMOptions scope(Settings{{"triton-test-no-such-option", "true"},
                                   {"triton-test-bool-option", "true"}});
  EXPECT_TRUE(boolOption.getValue());
}

TEST(LLVMOptions, IdenticalScopesShareTheRegistry) {
  ScopedLLVMOptions outer(Settings{{"triton-test-bool-option", "true"}});
  {
    ScopedLLVMOptions inner(Settings{{"triton-test-bool-option", "true"}});
    EXPECT_TRUE(boolOption.getValue());
    EXPECT_EQ(boolOption.getNumOccurrences(), 1);
  }
  // The outer scope still needs the override.
  EXPECT_TRUE(boolOption.getValue());
  EXPECT_EQ(boolOption.getNumOccurrences(), 1);
}

TEST(LLVMOptions, DifferentScopeWaitsForCurrentUsers) {
  auto outer = std::make_unique<ScopedLLVMOptions>(
      Settings{{"triton-test-bool-option", "true"}});
  BackgroundTask inner([] {
    ScopedLLVMOptions scope(Settings{{"triton-test-bool-option", "false"}});
    EXPECT_FALSE(boolOption.getValue());
    EXPECT_EQ(boolOption.getNumOccurrences(), 1);
  });
  EXPECT_FALSE(inner.finishedWithinGracePeriod());
  EXPECT_TRUE(boolOption.getValue());
  outer.reset();
  inner.join();
  EXPECT_FALSE(boolOption.getValue());
  EXPECT_EQ(boolOption.getNumOccurrences(), 0);
}

TEST(LLVMOptions, EmptyScopeWaitsForOverrides) {
  auto outer = std::make_unique<ScopedLLVMOptions>(
      Settings{{"triton-test-bool-option", "true"}});
  BackgroundTask defaults([] {
    ScopedLLVMOptions scope(Settings{});
    EXPECT_FALSE(boolOption.getValue());
    EXPECT_EQ(boolOption.getNumOccurrences(), 0);
  });
  EXPECT_FALSE(defaults.finishedWithinGracePeriod());
  outer.reset();
  defaults.join();
}

TEST(LLVMOptions, ExclusiveAccessWaitsForScopes) {
  // Even a scope without overrides keeps the registry stable.
  auto scope = std::make_unique<ScopedLLVMOptions>(Settings{});
  BackgroundTask exclusive([] {
    ExclusiveLLVMOptionAccess access;
    llvm::cl::ResetAllOptionOccurrences();
  });
  EXPECT_FALSE(exclusive.finishedWithinGracePeriod());
  scope.reset();
  exclusive.join();
}

TEST(LLVMOptions, ExclusiveAccessRestoresDefaults) {
  {
    ExclusiveLLVMOptionAccess access;
    bool parseFailed =
        boolOption.addOccurrence(0, "triton-test-bool-option", "true");
    EXPECT_FALSE(parseFailed);
    EXPECT_TRUE(boolOption.getValue());
  }
  EXPECT_FALSE(boolOption.getValue());
  EXPECT_EQ(boolOption.getNumOccurrences(), 0);
}

TEST(LLVMOptions, ScopesWaitForExclusiveAccess) {
  auto exclusive = std::make_unique<ExclusiveLLVMOptionAccess>();
  BackgroundTask scope([] {
    ScopedLLVMOptions scope(Settings{{"triton-test-bool-option", "true"}});
    EXPECT_TRUE(boolOption.getValue());
  });
  EXPECT_FALSE(scope.finishedWithinGracePeriod());
  exclusive.reset();
  scope.join();
  EXPECT_FALSE(boolOption.getValue());
}

#ifndef _WIN32
TEST(LLVMOptions, ForkedChildStartsWithFreshRegistry) {
  auto parentScope = std::make_unique<ScopedLLVMOptions>(
      Settings{{"triton-test-bool-option", "true"}});

  pid_t child = fork();
  ASSERT_NE(child, -1);
  if (child == 0) { // In the child process
    // This guard belongs to the parent's registry generation. Destroying it
    // must not underflow the child's freshly reset user count.
    parentScope.reset();
    bool succeeded = !boolOption.getValue();
    {
      ScopedLLVMOptions childScope(
          Settings{{"triton-test-bool-option", "true"}});
      succeeded &= boolOption.getValue();
    }
    succeeded &= !boolOption.getValue();
    _exit(succeeded ? 0 : 1);
  }

  // In the parent process
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  // The child reset must not affect the parent's registry.
  EXPECT_TRUE(boolOption.getValue());
  parentScope.reset();
  EXPECT_FALSE(boolOption.getValue());
}
#endif

} // namespace
