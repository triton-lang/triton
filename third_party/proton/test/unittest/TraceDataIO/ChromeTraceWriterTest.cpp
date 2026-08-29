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
  result->blockTraces[0].initTime = 0;
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
  EXPECT_DOUBLE_EQ(data["traceEvents"][0]["ts"], 0.0);
  EXPECT_DOUBLE_EQ(data["traceEvents"][1]["ts"], 0.1);
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
  EXPECT_EQ(pidCount["kernel2 Core121 CTA2"], 12);
  EXPECT_EQ(pidCount["kernel2 Core120 CTA1"], 12);
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
      result1->blockTraces[j].initTime = 0;
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
      result2->blockTraces[j].initTime = 10000000;
      result2->blockTraces[j].traces[i].uid = i;
      result2->blockTraces[j].traces[i].profileEvents[0].first->cycle = 1220000;
      result2->blockTraces[j].traces[i].profileEvents[0].second->cycle =
          1620000;
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
  EXPECT_DOUBLE_EQ(data["traceEvents"][0]["ts"], 0.0);
  EXPECT_DOUBLE_EQ(data["traceEvents"][0]["dur"], 400.0);
  EXPECT_EQ(data["traceEvents"][1]["cat"], "kernel1");
  EXPECT_EQ(data["traceEvents"][2]["cat"], "kernel2");
  EXPECT_DOUBLE_EQ(data["traceEvents"][2]["ts"], 10000.0);
  EXPECT_DOUBLE_EQ(data["traceEvents"][2]["dur"], 400.0);
}
TEST_F(ChromeTraceWriterTest, AsyncLinksAcrossWarps) {
  auto metadata = std::make_shared<KernelMetadata>();
  metadata->kernelName = "kernel";
  metadata->scopeName = {{1, "scope"}, {7, "async_copy"}};

  auto result = std::make_shared<CircularLayoutParserResult>();
  auto &block = result->blockTraces.emplace_back();
  block.blockId = 0;
  block.procId = 1;
  block.initTime = 1000;
  block.traces.resize(2);
  block.traces[0].uid = 2;
  block.traces[1].uid = 5;

  auto addScope = [&](auto &trace, uint64_t startCycle, uint64_t endCycle) {
    auto start = std::make_shared<CycleEntry>();
    start->cycle = startCycle;
    start->scopeId = 1;
    auto end = std::make_shared<CycleEntry>();
    end->cycle = endCycle;
    end->isStart = false;
    end->scopeId = 1;
    trace.profileEvents.emplace_back(start, end);
  };
  addScope(block.traces[0], 100, 125);
  addScope(block.traces[1], 150, 175);

  auto addLink = [&](uint64_t startCycle, uint64_t endCycle) {
    auto start = std::make_shared<CycleEntry>();
    start->cycle = startCycle;
    start->scopeId = 7;
    start->isAsync = true;
    auto end = std::make_shared<CycleEntry>();
    end->cycle = endCycle;
    end->isStart = false;
    end->scopeId = 7;
    end->isAsync = true;
    block.traces[0].asyncEvents.push_back(start);
    block.traces[1].asyncEvents.push_back(end);
    block.asyncLinks.push_back({{2, start}, {5, end}});
  };
  addLink(100, 150);
  addLink(200, 250);

  std::vector<KernelTrace> kernelTrace = {{result, metadata}};
  StreamChromeTraceWriter(kernelTrace, chromeTracePath).dump();
  auto data = readJsonTrace(chromeTracePath);

  ASSERT_EQ(data["traceEvents"].size(), 6);
  EXPECT_EQ(data["traceEvents"][0]["tid"], "warp 2 (line 0)");
  EXPECT_EQ(data["traceEvents"][1]["tid"], "warp 5 (line 0)");

  json flows = json::array();
  for (const auto &event : data["traceEvents"])
    if (event["cat"] == "flow")
      flows.push_back(event);
  ASSERT_EQ(flows.size(), 4);
  EXPECT_EQ(flows[0]["name"], "async_copy");
  EXPECT_EQ(flows[0]["ph"], "X");
  EXPECT_EQ(flows[1]["ph"], "X");
  EXPECT_TRUE(flows[0]["flow_out"]);
  EXPECT_TRUE(flows[1]["flow_in"]);
  EXPECT_EQ(flows[0]["bind_id"], flows[1]["bind_id"]);
  EXPECT_NE(flows[0]["bind_id"], flows[2]["bind_id"]);
  EXPECT_EQ(flows[0]["tid"], "warp 2 (line 0)");
  EXPECT_EQ(flows[1]["tid"], "warp 5 (line 0)");
  EXPECT_DOUBLE_EQ(flows[0]["ts"], 0.0);
  EXPECT_DOUBLE_EQ(flows[1]["ts"], 0.05);
  EXPECT_DOUBLE_EQ(flows[2]["ts"], 0.1);
  EXPECT_DOUBLE_EQ(flows[3]["ts"], 0.15);
  for (auto &event : flows)
    EXPECT_EQ(event["dur"], 0);
}

TEST_F(ChromeTraceWriterTest, AsyncEventsOnSameWarpUseDurationSlices) {
  auto metadata = std::make_shared<KernelMetadata>();
  metadata->kernelName = "kernel";
  metadata->scopeName = {
      {7, "async_copy[0]"},
      {8, "async_copy[1]"},
  };

  auto result = std::make_shared<CircularLayoutParserResult>();
  auto &block = result->blockTraces.emplace_back();
  block.blockId = 0;
  block.procId = 1;
  block.initTime = 1000;
  block.traces.resize(1);
  block.traces[0].uid = 2;

  auto addLink = [&](int scopeId, uint64_t startCycle, uint64_t endCycle) {
    auto start = std::make_shared<CycleEntry>();
    start->cycle = startCycle;
    start->scopeId = scopeId;
    start->isAsync = true;
    auto end = std::make_shared<CycleEntry>();
    end->cycle = endCycle;
    end->isStart = false;
    end->scopeId = scopeId;
    end->isAsync = true;
    block.traces[0].asyncEvents.push_back(start);
    block.traces[0].asyncEvents.push_back(end);
    block.asyncLinks.push_back({{2, start}, {2, end}});
  };
  addLink(7, 100, 200);
  addLink(8, 150, 250);
  addLink(7, 200, 300);

  std::vector<KernelTrace> kernelTrace = {{result, metadata}};
  StreamChromeTraceWriter(kernelTrace, chromeTracePath).dump();
  auto data = readJsonTrace(chromeTracePath);

  ASSERT_EQ(data["traceEvents"].size(), 3);
  EXPECT_EQ(data["traceEvents"][0]["ph"], "X");
  EXPECT_EQ(data["traceEvents"][0]["cat"], "async");
  EXPECT_EQ(data["traceEvents"][0]["tid"], "warp 2 (line 0)");
  EXPECT_DOUBLE_EQ(data["traceEvents"][0]["ts"], 0.0);
  EXPECT_DOUBLE_EQ(data["traceEvents"][0]["dur"], 0.1);
  EXPECT_EQ(data["traceEvents"][1]["tid"], "warp 2 (line 1)");
  EXPECT_DOUBLE_EQ(data["traceEvents"][1]["ts"], 0.05);
  EXPECT_DOUBLE_EQ(data["traceEvents"][1]["dur"], 0.1);
  EXPECT_EQ(data["traceEvents"][2]["tid"], "warp 2 (line 0)");
  EXPECT_DOUBLE_EQ(data["traceEvents"][2]["ts"], 0.1);
  EXPECT_DOUBLE_EQ(data["traceEvents"][2]["dur"], 0.1);
  for (auto &event : data["traceEvents"]) {
    EXPECT_FALSE(event.contains("bind_id"));
    EXPECT_FALSE(event.contains("flow_in"));
    EXPECT_FALSE(event.contains("flow_out"));
  }
}
