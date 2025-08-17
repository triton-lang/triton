#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <cstdint>
  
// Small types to model metadata used by the runtime reflection parser.
//
// Some toolchains / IntelliSense configurations may not enable C++17. To avoid
// relying on std::optional in this lightweight unit test, we represent optional
// fields with explicit presence flags. This keeps the test compatible with
// older compilers and with tools that don't parse C++17 constructs.
struct ArgMeta {
  int buffer_index = -1;
  bool has_buffer_index = false;
  std::string type;
  bool has_type = false;
};

// Determine binding index using positional binding with optional metadata override.
// Mirrors the behavior in third_party/metal/backend/runtime.py where metadata may
// provide 'buffer_index' and any parsing errors fall back to positional binding.
static int determine_bind_index(size_t pos, const std::vector<ArgMeta> &meta) {
  if (!meta.empty()) {
    if (pos < meta.size()) {
      auto &am = meta[pos];
      if (am.has_buffer_index) {
        return am.buffer_index;
      }
    }
  }
  return static_cast<int>(pos);
}

TEST(ArgumentBinding, PositionalByDefault) {
  std::vector<ArgMeta> meta;  // empty metadata
  EXPECT_EQ(determine_bind_index(0, meta), 0);
  EXPECT_EQ(determine_bind_index(1, meta), 1);
  EXPECT_EQ(determine_bind_index(5, meta), 5);
}

TEST(ArgumentBinding, ReflectionOverridesPosition) {
  std::vector<ArgMeta> meta(3);
  meta[0].buffer_index = 2;
  meta[0].has_buffer_index = true;
  meta[1].buffer_index = 0;
  meta[1].has_buffer_index = true;
  // third arg has no override
  EXPECT_EQ(determine_bind_index(0, meta), 2);
  EXPECT_EQ(determine_bind_index(1, meta), 0);
  EXPECT_EQ(determine_bind_index(2, meta), 2); // falls back to positional (2)
}

TEST(ArgumentBinding, OutOfRangeMetadataFallsBack) {
  std::vector<ArgMeta> meta(1);
  meta[0].buffer_index = 7;
  meta[0].has_buffer_index = true;
  // asking for position 3 -> no metadata entry -> positional binding
  EXPECT_EQ(determine_bind_index(3, meta), 3);
}

TEST(ArgumentBinding, PODSizesAndBytes) {
  // The runtime packs ints as 64-bit and floats as 64-bit (struct.pack 'q' / 'd').
  static_assert(sizeof(long long) >= 8, "expected 64-bit storage for integer packing");
  static_assert(sizeof(double) == 8, "double must be 8 bytes");
  // Sanity checks used by tests that assert packing sizes
  EXPECT_EQ(sizeof(int64_t), 8);
  EXPECT_EQ(sizeof(double), 8);

  // Simulate a small byte buffer argument and ensure length detection works.
  std::string bytes = "abcd";
  EXPECT_EQ(bytes.size(), 4u);
  std::vector<uint8_t> vec(bytes.begin(), bytes.end());
  EXPECT_EQ(vec.size(), 4u);
  // Ensure converting from memory-like objects would preserve length
  EXPECT_EQ(static_cast<size_t>(bytes.size()), vec.size());
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
