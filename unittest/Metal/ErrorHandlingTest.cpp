#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <optional>

// Simulated error types used by the Metal backend runtime.
struct CompileError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct ToolMissingError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct RuntimeExecutionError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Simulated API functions
static void check_toolchain(bool present) {
  if (!present) throw ToolMissingError("required tool not found: xcrun");
}

static void compile_library(bool succeed) {
  if (!succeed) throw CompileError("metal compiler failed: syntax error");
}

static void run_kernel(bool succeed) {
  if (!succeed) throw RuntimeExecutionError("kernel execution failed: out-of-bound");
}

TEST(ErrorHandling, MissingToolProducesToolMissingError) {
  EXPECT_THROW(check_toolchain(false), ToolMissingError);
  try {
    check_toolchain(false);
  } catch (const std::exception &e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("xcrun"), std::string::npos);
  }
}

TEST(ErrorHandling, CompileFailurePropagatesCompileError) {
  EXPECT_THROW(compile_library(false), CompileError);
  try {
    compile_library(false);
  } catch (const CompileError &e) {
    EXPECT_NE(std::string(e.what()).find("syntax"), std::string::npos);
  }
}

TEST(ErrorHandling, RuntimeFailurePropagatesRuntimeExecutionError) {
  EXPECT_THROW(run_kernel(false), RuntimeExecutionError);
  try {
    run_kernel(false);
  } catch (const RuntimeExecutionError &e) {
    EXPECT_NE(std::string(e.what()).find("out-of-bound"), std::string::npos);
  }
}

TEST(ErrorHandling, SuccessfulPathsDoNotThrow) {
  EXPECT_NO_THROW(check_toolchain(true));
  EXPECT_NO_THROW(compile_library(true));
  EXPECT_NO_THROW(run_kernel(true));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}