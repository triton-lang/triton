#include "TraceDataIO/CircularLayoutParser.h"
#include <cstdlib>
#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

using namespace proton;

class CircularLayoutParserTest : public ::testing::Test {
public:
  explicit CircularLayoutParserTest(const std::string &kernel = "")
      : kernel(kernel) {}

  void SetUp() override {
    if (!kernel.empty()) {
      output = PROTON_TEST_UTIL_PATH;
      output += "/" + kernel + ".bin";
    }
  }

  void TearDown() override {}

  ByteSpan getBuffer(std::string binPath) {
    std::ifstream file(binPath, std::ios::binary);

    if (!file) {
      std::cerr << "Cannot open file!" << std::endl;
      return ByteSpan(nullptr, 0);
    }

    // Get file size
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    testData.resize(size);

    // Read the data
    if (!file.read(reinterpret_cast<char *>(testData.data()), size)) {
      std::cerr << "Error reading file!" << std::endl;
      return ByteSpan(nullptr, 0);
    }
    return ByteSpan(testData.data(), size);
  }

protected:
  CircularLayoutParserConfig config;
  std::vector<uint8_t> testData;
  std::string kernel;
  std::string output;
};

TEST_F(CircularLayoutParserTest, WrongPreamble) {
  config.numBlocks = 1;
  config.uidVec = {0};
  testData = {0x78, 0x56, 0x34, 0x12, 0x01, 0x00,
              0x00, 0x80, 0xFF, 0xFF, 0xFF, 0xFF};
  auto buffer = ByteSpan(testData.data(), testData.size());
  auto parser = CircularLayoutParser(buffer, config);
  EXPECT_THROW(parser.parse(), ParserException);
}

TEST_F(CircularLayoutParserTest, SingleEvent) {
  // clang-format off
  testData = {0xef, 0xbe, 0xad, 0xde,
              0x01, 0x00, 0x00, 0x00,
              0x03, 0x00, 0x00, 0x00,
              0x10, 0x00, 0x00, 0x00,
              0xff, 0x00, 0x00, 0x00,
              0x00, 0x00, 0x00, 0x02,
              0x00, 0x10, 0x00, 0x00,
              0x00, 0x00, 0x00, 0x82,
              0x00, 0x20, 0x00, 0x00};
  // clang-format on
  config.numBlocks = 1;
  config.totalUnits = 1;
  config.scratchMemSize = testData.size();
  config.uidVec = {0};
  auto buffer = ByteSpan(testData.data(), testData.size());
  auto parser = CircularLayoutParser(buffer, config);
  parser.parse();
  auto result = parser.getResult();
  EXPECT_EQ(result->blockTraces.size(), 1);
  EXPECT_EQ(result->blockTraces[0].blockId, 1);
  EXPECT_EQ(result->blockTraces[0].procId, 3);
  EXPECT_EQ(result->blockTraces[0].traces[0].count, 255);
  EXPECT_EQ(result->blockTraces[0].traces[0].uid, 0);
  EXPECT_EQ(result->blockTraces[0].traces[0].profileEvents.size(), 1);
  auto &event = result->blockTraces[0].traces[0].profileEvents[0];
  EXPECT_EQ(event.first->scopeId, 4);
  EXPECT_EQ(event.second->scopeId, 4);
  EXPECT_EQ(event.first->isStart, true);
  EXPECT_EQ(event.second->isStart, false);
  EXPECT_EQ(event.first->cycle, 4096);
  EXPECT_EQ(event.second->cycle, 8192);
}

TEST_F(CircularLayoutParserTest, StartAfterStart) {
  // clang-format off
  testData = {0xef, 0xbe, 0xad, 0xde,
              0x01, 0x00, 0x00, 0x00,
              0x03, 0x00, 0x00, 0x00,
              0x10, 0x00, 0x00, 0x00,
              0xff, 0x00, 0x00, 0x00,
              0x04, 0x00, 0x00, 0x00,
              0x00, 0x10, 0x00, 0x00,
              0x04, 0x00, 0x00, 0x00,
              0x00, 0x20, 0x00, 0x00};
  // clang-format on
  config.numBlocks = 1;
  config.totalUnits = 1;
  config.scratchMemSize = testData.size();
  config.uidVec = {0};
  auto buffer = ByteSpan(testData.data(), testData.size());
  auto parser = CircularLayoutParser(buffer, config);
  parser.parse();
  auto result = parser.getResult();
  EXPECT_EQ(result->blockTraces[0].traces[0].profileEvents.size(), 0);
}

TEST_F(CircularLayoutParserTest, MultipleSegment) {
  // clang-format off
  testData = {0xef, 0xbe, 0xad, 0xde,
              0x01, 0x00, 0x00, 0x00,
              0x03, 0x00, 0x00, 0x00,
              0x30, 0x00, 0x00, 0x00,
              0xff, 0x00, 0x00, 0x00,
              0xff, 0x00, 0x00, 0x00,
              0xff, 0x00, 0x00, 0x00,
              0x00, 0x00, 0x00, 0x00,
              0x00, 0x10, 0x00, 0x00,
              0x00, 0x00, 0x00, 0x80,
              0x00, 0x20, 0x00, 0x00,
              0x00, 0x00, 0x00, 0x00,
              0x00, 0x10, 0x00, 0x00,
              0x00, 0x00, 0x00, 0x80,
              0x00, 0x20, 0x00, 0x00,
              0x00, 0x00, 0x00, 0x00,
              0x00, 0x10, 0x00, 0x00,
              0x00, 0x00, 0x00, 0x80,
              0x00, 0x20, 0x00, 0x00,
              0xff, 0xff, 0xff, 0xff,
              0xff, 0xff, 0xff, 0xff};
  // clang-format on
  config.numBlocks = 1;
  config.totalUnits = 3;
  config.scratchMemSize = testData.size();
  config.uidVec = {0, 1, 2};
  auto buffer = ByteSpan(testData.data(), testData.size());
  auto parser = CircularLayoutParser(buffer, config);
  parser.parse();
  auto result = parser.getResult();
  EXPECT_EQ(result->blockTraces[0].traces.size(), 3);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(result->blockTraces[0].traces[i].profileEvents.size(), 1);
    EXPECT_EQ(result->blockTraces[0].traces[i].profileEvents[0].first->cycle,
              4096);
    EXPECT_EQ(result->blockTraces[0].traces[i].profileEvents[0].second->cycle,
              8192);
  }
}

class CLParserSeqTraceTest : public CircularLayoutParserTest {
public:
  CLParserSeqTraceTest() : CircularLayoutParserTest("seq") {}
};

TEST_F(CLParserSeqTraceTest, Trace) {
  auto buffer = getBuffer(output);
  auto result = proton::readCircularLayoutTrace(buffer);
  EXPECT_EQ(result->blockTraces.size(), 2);
  EXPECT_EQ(result->blockTraces[1].blockId, 1);
  EXPECT_EQ(result->blockTraces[0].traces.size(), 4);
  EXPECT_EQ(result->blockTraces[0].traces[0].count, 12);
  EXPECT_EQ(result->blockTraces[0].traces[3].profileEvents.size(), 3);
}

class CLParserLoopTraceTest : public CircularLayoutParserTest {
public:
  CLParserLoopTraceTest() : CircularLayoutParserTest("loop") {}
};

TEST_F(CLParserLoopTraceTest, Trace) {
  auto buffer = getBuffer(output);
  auto result = proton::readCircularLayoutTrace(buffer);
  EXPECT_EQ(result->blockTraces.size(), 1);
  EXPECT_EQ(result->blockTraces[0].traces.size(), 4);
  EXPECT_EQ(result->blockTraces[0].traces[0].count, 80);
  EXPECT_EQ(result->blockTraces[0].traces[3].profileEvents.size(), 4);
}

TEST_F(CircularLayoutParserTest, TimeShift) {
  // clang-format off
  testData = {0xef, 0xbe, 0xad, 0xde,
              0x01, 0x00, 0x00, 0x00,
              0x03, 0x00, 0x00, 0x00,
              0x20, 0x00, 0x00, 0x00,
              0xff, 0x00, 0x00, 0x00,
              0x00, 0x00, 0x00, 0x00,
              0x21, 0x00, 0x00, 0x00,
              0x00, 0x00, 0x00, 0x01,
              0x36, 0x00, 0x00, 0x00,
              0x00, 0x00, 0x00, 0x80,
              0x46, 0x00, 0x00, 0x00,
              0x00, 0x00, 0x00, 0x81,
              0x64, 0x00, 0x00, 0x00,};
  // clang-format on
  config.numBlocks = 1;
  config.totalUnits = 1;
  config.scratchMemSize = testData.size();
  config.uidVec = {0};
  config.device.type = DeviceType::CUDA;
  auto buffer = ByteSpan(testData.data(), testData.size());
  auto parser = CircularLayoutParser(buffer, config);
  parser.parse();
  auto result = parser.getResult();
  auto &event0 = result->blockTraces[0].traces[0].profileEvents[0];
  auto &event1 = result->blockTraces[0].traces[0].profileEvents[1];
  EXPECT_EQ(event0.first->cycle, 33);
  EXPECT_EQ(event0.second->cycle, 70);
  EXPECT_EQ(event1.first->cycle, 54);
  EXPECT_EQ(event1.second->cycle, 100);

  const uint64_t cost = getTimeShiftCost(config);
  timeShift(cost, result);

  EXPECT_EQ(event0.first->cycle, 26);
  EXPECT_EQ(event0.second->cycle, 49);
  EXPECT_EQ(event1.first->cycle, 40);
  EXPECT_EQ(event1.second->cycle, 72);
}
