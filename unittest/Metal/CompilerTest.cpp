#include <gtest/gtest.h>
#include <regex>
#include <string>
#include <vector>
#include <optional>
#include <map>

// Minimal reflection-like parser used by tests to mirror python/compiler.py behaviour.
// This is intentionally conservative and only implements the features required for tests.
struct ArgInfo {
  std::string raw;
  std::optional<std::string> type;
  std::optional<std::string> name;
  std::optional<int> buffer_index;
  std::optional<std::string> address_space;
};

struct KernelInfo {
  std::string name;
  std::vector<ArgInfo> args;
};

// Parse kernels from Metal source. Returns list of KernelInfo.
static std::vector<KernelInfo> parse_kernels(const std::string &src) {
  std::vector<KernelInfo> kernels;
  std::regex kernel_re(R"(\bkernel\b(?:\s+\w+)?\s+([A-Za-z_]\w*)\s*\(([^)]*)\))", std::regex::ECMAScript);
  std::regex buffer_idx_re(R"(\[\[\s*buffer\s*\(\s*(\d+)\s*\)\s*\]\])", std::regex::ECMAScript);
  auto begin = std::sregex_iterator(src.begin(), src.end(), kernel_re);
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end; ++it) {
    std::smatch m = *it;
    KernelInfo k;
    k.name = m[1].str();
    std::string params = m[2].str();
    if (!params.empty()) {
      // split by commas (simple)
      size_t pos = 0;
      while (pos < params.size()) {
        size_t comma = params.find(',', pos);
        std::string raw = params.substr(pos, (comma == std::string::npos) ? std::string::npos : comma - pos);
        // trim
        auto l = raw.find_first_not_of(" \t\n\r");
        auto r = raw.find_last_not_of(" \t\n\r");
        if (l == std::string::npos) {
          // empty
        } else {
          raw = raw.substr(l, r - l + 1);
          ArgInfo ai;
          ai.raw = raw;
          std::smatch bufm;
          if (std::regex_search(raw, bufm, buffer_idx_re)) {
            ai.buffer_index = std::stoi(bufm[1].str());
          }
          // remove any [[ ... ]] attributes
          std::string cleaned = std::regex_replace(raw, std::regex(R"(\[\[.*?\]\])"), "");
          // normalize whitespace
          cleaned = std::regex_replace(cleaned, std::regex(R"(\s+)"), " ");
          // split tokens
          std::vector<std::string> tokens;
          {
            std::istringstream iss(cleaned);
            std::string tok;
            while (iss >> tok) tokens.push_back(tok);
          }
          if (tokens.size() >= 2) {
            ai.name = tokens.back();
            std::string t;
            for (size_t i = 0; i + 1 < tokens.size(); ++i) {
              if (i) t += " ";
              t += tokens[i];
            }
            ai.type = t;
          } else if (tokens.size() == 1) {
            ai.type = tokens[0];
            ai.name = std::nullopt;
          } else {
            ai.name = std::nullopt;
            ai.type = std::nullopt;
          }
          // address space inference
          if (ai.type) {
            if (ai.type->find("device") != std::string::npos) ai.address_space = "device";
            else if (ai.type->find("threadgroup") != std::string::npos) ai.address_space = "threadgroup";
            else if (ai.type->find("constant") != std::string::npos) ai.address_space = "constant";
          }
          k.args.push_back(std::move(ai));
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
    }
    kernels.push_back(std::move(k));
  }
  return kernels;
}

TEST(MetalCompiler, ParseSimpleKernel) {
  const std::string src = R"(
    kernel void add(device float *in [[ buffer(0) ]], device float *out [[ buffer(1) ]], uint n) {
    }
  )";
  auto kernels = parse_kernels(src);
  ASSERT_EQ(kernels.size(), 1u);
  EXPECT_EQ(kernels[0].name, "add");
  ASSERT_EQ(kernels[0].args.size(), 3u);
  EXPECT_EQ(kernels[0].args[0].buffer_index.value_or(-1), 0);
  EXPECT_EQ(kernels[0].args[0].address_space.value_or(""), "device");
  EXPECT_EQ(kernels[0].args[0].name.value_or(""), "in");
  EXPECT_EQ(kernels[0].args[1].buffer_index.value_or(-1), 1);
  EXPECT_EQ(kernels[0].args[1].name.value_or(""), "out");
  EXPECT_EQ(kernels[0].args[2].buffer_index.has_value(), false);
  EXPECT_EQ(kernels[0].args[2].type.value_or(""), "uint");
  EXPECT_FALSE(kernels[0].args[2].address_space.has_value());
}

TEST(MetalCompiler, HandlesNoArgsAndUnnamed) {
  const std::string src = R"(
    kernel void foo() { }
    kernel void bar(device float *data [[ buffer(2) ]]) {}
    kernel void weird(float) {}
  )";
  auto kernels = parse_kernels(src);
  ASSERT_EQ(kernels.size(), 3u);
  EXPECT_EQ(kernels[0].name, "foo");
  EXPECT_EQ(kernels[1].name, "bar");
  EXPECT_EQ(kernels[1].args.size(), 1u);
  EXPECT_EQ(kernels[1].args[0].buffer_index.value_or(-1), 2);
  EXPECT_EQ(kernels[2].args.size(), 1u);
  // unnamed param should produce no name but a type
  EXPECT_TRUE(kernels[2].args[0].name == std::nullopt);
  EXPECT_TRUE(kernels[2].args[0].type.has_value());
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}