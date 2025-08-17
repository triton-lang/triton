#include <gtest/gtest.h>
#include <string>
#include <map>
#include <stdexcept>
#include <vector>

// Minimal simulated Metal library handle for unit tests. Tests only the
// behaviours required by the runtime tests: stub reporting, pipeline cache,
// and kernel lookup error handling.
class FakeMetalLibrary {
public:
  FakeMetalLibrary(bool stub_mode, const std::vector<std::string> &kernels = {})
      : stub_(stub_mode), available_kernels_(kernels), next_id_(1) {}

  bool is_stub() const { return stub_; }

  // Create or return cached pipeline id for kernel name.
  int get_or_create_pipeline(const std::string &kernel_name) {
    auto it = pipeline_cache_.find(kernel_name);
    if (it != pipeline_cache_.end()) return it->second;
    // Simulate pipeline creation. In stub mode we still create a placeholder id.
    int id = next_id_++;
    pipeline_cache_[kernel_name] = id;
    return id;
  }

  // Lookup kernel by name, throw if not found (simulates runtime lookup error).
  void ensure_kernel_exists(const std::string &kernel_name) const {
    for (const auto &k : available_kernels_) {
      if (k == kernel_name) return;
    }
    throw std::runtime_error("kernel not found: " + kernel_name);
  }

  size_t pipeline_cache_size() const { return pipeline_cache_.size(); }

private:
  bool stub_;
  std::vector<std::string> available_kernels_;
  std::map<std::string, int> pipeline_cache_;
  int next_id_;
};

TEST(MetalRuntime, StubHandleReportsStubMode) {
  FakeMetalLibrary stub_lib(true, {});
  EXPECT_TRUE(stub_lib.is_stub());
  // In stub mode we can still create pipelines (placeholders).
  int p = stub_lib.get_or_create_pipeline("k");
  EXPECT_GT(p, 0);
}

TEST(MetalRuntime, PipelineCacheReturnsSameId) {
  FakeMetalLibrary lib(false, {"add", "mul"});
  int p1 = lib.get_or_create_pipeline("add");
  int p2 = lib.get_or_create_pipeline("add");
  EXPECT_EQ(p1, p2);
  // Different kernel -> different id
  int p3 = lib.get_or_create_pipeline("mul");
  EXPECT_NE(p1, p3);
  EXPECT_EQ(lib.pipeline_cache_size(), 2u);
}

TEST(MetalRuntime, KernelLookupMissingThrows) {
  FakeMetalLibrary lib(false, {"exist"});
  EXPECT_NO_THROW(lib.ensure_kernel_exists("exist"));
  EXPECT_THROW(lib.ensure_kernel_exists("does_not_exist"), std::runtime_error);
  try {
    lib.ensure_kernel_exists("does_not_exist");
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("kernel not found"), std::string::npos);
  }
}

TEST(MetalRuntime, StubModeStillCachesPipelinesSeparately) {
  FakeMetalLibrary stub_lib(true, {"k1"});
  int a = stub_lib.get_or_create_pipeline("k1");
  int b = stub_lib.get_or_create_pipeline("k1");
  EXPECT_EQ(a, b);
  EXPECT_EQ(stub_lib.pipeline_cache_size(), 1u);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}