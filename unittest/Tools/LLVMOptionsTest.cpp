#include "triton/Tools/LLVMOptions.h"

#include "llvm/Support/CommandLine.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

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

} // namespace
