#include "TraceDataIO/EntryDecoder.h"
#include "TraceDataIO/TraceWriter.h"
#include "nlohmann/json.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

using json = nlohmann::json;
using namespace proton;

class ChromeTraceWriterTest : public ::testing::Test {
public:
  void SetUp() override {}

  void TearDown() override {
    try {
      std::filesystem::remove_all(chromeTracePath);
    } catch (const std::filesystem::filesystem_error &e) {
      std::cerr << "Error cleaning up test trace files: " << e.what()
                << std::endl;
    }
  }

  void printJsonTrace(json data) { std::cout << data.dump(4) << std::endl; }

  json readJsonTrace(const std::string &path) {
    std::ifstream file(path);

    if (!file.is_open()) {
      std::cerr << "Failed to open chrome trace file!" << std::endl;
      return json();
    }

    json data;
    try {
      data = json::parse(file);
    } catch (json::parse_error &e) {
      std::cerr << "Error parsing JSON: " << e.what() << std::endl;
      data = json();
    }
    file.close();
    return data;
  }

  std::shared_ptr<CircularLayoutParserResult>
  createDefaultResult(int numBlocks, int numTraces, int numEvents) {
    auto result = std::make_shared<CircularLayoutParserResult>();
    result->blockTraces.resize(numBlocks);
    for (int i = 0; i < numBlocks; i++) {
      result->blockTraces[i].traces.resize(numTraces);
      for (int j = 0; j < numTraces; j++) {
        result->blockTraces[i].traces[j].profileEvents.resize(numEvents);
        for (int k = 0; k < numEvents; k++) {
          result->blockTraces[i].traces[j].profileEvents[k].first =
              std::make_shared<CycleEntry>();
          result->blockTraces[i].traces[j].profileEvents[k].second =
              std::make_shared<CycleEntry>();
        }
      }
    }
    return result;
  }

protected:
  std::string chromeTracePath = "chrome_trace.json";
};

TEST_F(ChromeTraceWriterTest, SingleBlock) {
  auto metadata = std::make_shared<KernelMetadata>();
  metadata->kernelName = "kernel1";
  metadata->scopeName = {{1, "s1"}, {2, "s2"}};

  auto result = createDefaultResult(1, 1, metadata->scopeName.size());
  result->blockTraces[0].blockId = 1;
  result->blockTraces[0].procId = 120;
  result->blockTraces[0].traces[0].uid = 2;
  result->blockTraces[0].traces[0].profileEvents[0].first->cycle = 122;
  result->blockTraces[0].traces[0].profileEvents[0].second->cycle = 162;
  result->blockTraces[0].traces[0].profileEvents[0].first->scopeId = 1;
  result->blockTraces[0].traces[0].profileEvents[0].second->scopeId = 1;
  result->blockTraces[0].traces[0].profileEvents[1].first->cycle = 222;
  result->blockTraces[0].traces[0].profileEvents[1].second->cycle = 262;
  result->blockTraces[0].traces[0].profileEvents[1].first->scopeId = 7;
  result->blockTraces[0].traces[0].profileEvents[1].second->scopeId = 7;
  std::vector<KernelTrace> kerneltrace = {std::make_pair(result, metadata)};
  auto writer = StreamChromeTraceWriter(kerneltrace, chromeTracePath);
  writer.dump();

  auto data = readJsonTrace(chromeTracePath);
  EXPECT_EQ(data.empty(), false);
  EXPECT_EQ(data["displayTimeUnit"], "ns");
  EXPECT_EQ(data["traceEvents"].size(), 2);
  EXPECT_EQ(data["traceEvents"][0]["name"], "s1");
  EXPECT_EQ(data["traceEvents"][1]["name"], "scope_7");
  EXPECT_EQ(data["traceEvents"][0]["ts"], 0.0);
  EXPECT_EQ(data["traceEvents"][1]["ts"], 0.1);
}

TEST_F(ChromeTraceWriterTest, MultiBlockMultiWarp) {
  auto metadata = std::make_shared<KernelMetadata>();
  metadata->kernelName = "kernel2";
  metadata->scopeName = {{1, "s1"}, {2, "s2"}, {3, "s3"}, {4, "s4"}};

  auto result = createDefaultResult(2, 3, metadata->scopeName.size());

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      result->blockTraces[j].blockId = 1 + j;
      result->blockTraces[j].procId = 120 + j;
      result->blockTraces[j].traces[i].uid = i;
      result->blockTraces[j].traces[i].profileEvents[0].first->cycle = 122;
      result->blockTraces[j].traces[i].profileEvents[0].second->cycle = 162;
      result->blockTraces[j].traces[i].profileEvents[0].first->scopeId = 1;
      result->blockTraces[j].traces[i].profileEvents[0].second->scopeId = 1;
      result->blockTraces[j].traces[i].profileEvents[1].first->cycle = 142;
      result->blockTraces[j].traces[i].profileEvents[1].second->cycle = 182;
      result->blockTraces[j].traces[i].profileEvents[1].first->scopeId = 2;
      result->blockTraces[j].traces[i].profileEvents[1].second->scopeId = 2;
      result->blockTraces[j].traces[i].profileEvents[2].first->cycle = 172;
      result->blockTraces[j].traces[i].profileEvents[2].second->cycle = 200;
      result->blockTraces[j].traces[i].profileEvents[2].first->scopeId = 3;
      result->blockTraces[j].traces[i].profileEvents[2].second->scopeId = 3;
      result->blockTraces[j].traces[i].profileEvents[3].first->cycle = 183;
      result->blockTraces[j].traces[i].profileEvents[3].second->cycle = 210;
      result->blockTraces[j].traces[i].profileEvents[3].first->scopeId = 4;
      result->blockTraces[j].traces[i].profileEvents[3].second->scopeId = 4;
    }
  }
  std::vector<KernelTrace> kerneltrace = {std::make_pair(result, metadata)};
  auto writer = StreamChromeTraceWriter(kerneltrace, chromeTracePath);
  writer.dump();

  auto data = readJsonTrace(chromeTracePath);

  EXPECT_EQ(data.empty(), false);
  EXPECT_EQ(data["traceEvents"].size(), 24);
  std::map<std::string, int> pidCount;
  std::map<std::string, int> tidCount;
  for (int i = 0; i < 24; i++) {
    pidCount[data["traceEvents"][i]["pid"]] += 1;
    tidCount[data["traceEvents"][i]["tid"]] += 1;
  }
  EXPECT_EQ(
      pidCount["kernel2 Core121 CTA2 [measure in clock cycle (assume 1GHz)]"],
      12);
  EXPECT_EQ(
      pidCount["kernel2 Core120 CTA1 [measure in clock cycle (assume 1GHz)]"],
      12);
  EXPECT_EQ(tidCount["warp 0 (line 0)"], 4);
  EXPECT_EQ(tidCount["warp 0 (line 1)"], 4);
  EXPECT_EQ(tidCount["warp 1 (line 0)"], 4);
  EXPECT_EQ(tidCount["warp 1 (line 1)"], 4);
  EXPECT_EQ(tidCount["warp 2 (line 0)"], 4);
  EXPECT_EQ(tidCount["warp 2 (line 1)"], 4);
}

TEST_F(ChromeTraceWriterTest, MultiKernel) {
  auto metadata1 = std::make_shared<KernelMetadata>();
  metadata1->kernelName = "kernel1";
  metadata1->scopeName = {{1, "s1"}};
  auto result1 = createDefaultResult(1, 2, metadata1->scopeName.size());

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 1; j++) {
      result1->blockTraces[j].blockId = j;
      result1->blockTraces[j].procId = j;
      result1->blockTraces[j].traces[i].uid = i;
      result1->blockTraces[j].traces[i].profileEvents[0].first->cycle = 1220000;
      result1->blockTraces[j].traces[i].profileEvents[0].second->cycle =
          1620000;
      result1->blockTraces[j].traces[i].profileEvents[0].first->scopeId = 1;
      result1->blockTraces[j].traces[i].profileEvents[0].second->scopeId = 1;
    }
  }

  auto metadata2 = std::make_shared<KernelMetadata>();
  metadata2->kernelName = "kernel2";
  metadata2->scopeName = {{1, "s1"}};
  auto result2 = createDefaultResult(2, 1, metadata2->scopeName.size());

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      result2->blockTraces[j].blockId = j;
      result2->blockTraces[j].procId = j;
      result2->blockTraces[j].traces[i].uid = i;
      result2->blockTraces[j].traces[i].profileEvents[0].first->cycle =
          2 * kKernelTimeGap + 1220000;
      result2->blockTraces[j].traces[i].profileEvents[0].second->cycle =
          2 * kKernelTimeGap + 1620000;
      result2->blockTraces[j].traces[i].profileEvents[0].first->scopeId = 1;
      result2->blockTraces[j].traces[i].profileEvents[0].second->scopeId = 1;
    }
  }
  std::vector<KernelTrace> kerneltrace = {std::make_pair(result1, metadata1),
                                          std::make_pair(result2, metadata2)};
  auto writer = StreamChromeTraceWriter(kerneltrace, chromeTracePath);
  writer.dump();

  auto data = readJsonTrace(chromeTracePath);

  EXPECT_EQ(data.empty(), false);
  EXPECT_EQ(data["traceEvents"][0]["cat"], "kernel1");
  EXPECT_EQ(data["traceEvents"][0]["ts"], 0.0);
  EXPECT_EQ(data["traceEvents"][0]["dur"], 400.0);
  EXPECT_EQ(data["traceEvents"][1]["cat"], "kernel1");
  EXPECT_EQ(data["traceEvents"][2]["cat"], "kernel2");
  EXPECT_EQ(data["traceEvents"][2]["ts"], 10000.0);
  EXPECT_EQ(data["traceEvents"][2]["dur"], 400.0);
}
