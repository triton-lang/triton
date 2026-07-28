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
  testData = {
      // header
      0xef, 0xbe, 0xad, 0xde, // preamble
      0x01, 0x00, 0x00, 0x00, // program id
      0x03, 0x00, 0x00, 0x00, // hw id
      0x10, 0x00, 0x00, 0x00, // buf size
      0xef, 0xcd, 0xab, 0x89, // initial time
      0x67, 0x45, 0x23, 0x01, //
      0x10, 0x32, 0x54, 0x76, // pre-final time
      0x98, 0xba, 0xdc, 0xfe, //
      0x08, 0x07, 0x06, 0x05, // post-final time
      0x04, 0x03, 0x02, 0x01, //
      // num events
      0xff, 0x00, 0x00, 0x00,
      // profiled data
      0x00, 0x00, 0x00, 0x02, // start
      0x00, 0x10, 0x00, 0x00, //
      0x00, 0x00, 0x00, 0x82, // end
      0x00, 0x20, 0x00, 0x00, //
  };
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
  EXPECT_EQ(result->blockTraces[0].initTime, 0x0123456789abcdef);
  EXPECT_EQ(result->blockTraces[0].preFinalTime, 0xfedcba9876543210);
  EXPECT_EQ(result->blockTraces[0].postFinalTime, 0x0102030405060708);
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

TEST_F(CircularLayoutParserTest, MetricAndMarker) {
  auto append32 = [&](uint32_t word) {
    for (int i = 0; i < 4; ++i)
      testData.push_back((word >> (i * 8)) & 0xff);
  };
  auto append64 = [&](uint64_t word) {
    append32(word);
    append32(word >> 32);
  };

  append32(kPreamble);
  append32(1);  // block id
  append32(3);  // processor id
  append32(32); // buffer size
  append64(10); // init
  append64(20); // pre-final
  append64(30); // post-final
  append32(8);  // four two-word records
  append32(4u << 16); // metric-bearing scope start
  append32(100);
  append32((3u << 20) | (4u << 16)); // I32 metric extension
  append32(static_cast<uint32_t>(-7));
  append32(1u << 31); // scope end
  append32(200);
  append32((1u << 23) | (2u << 20)); // marker
  append32(150);

  config.numBlocks = 1;
  config.totalUnits = 1;
  config.scratchMemSize = testData.size();
  config.uidVec = {0};
  auto buffer = ByteSpan(testData.data(), testData.size());
  CircularLayoutParser parser(buffer, config);
  parser.parse();

  auto &trace = parser.getResult()->blockTraces[0].traces[0];
  ASSERT_EQ(trace.profileEvents.size(), 1);
  ASSERT_TRUE(trace.profileEvents[0].first->metric);
  EXPECT_EQ(
      std::get<int64_t>(*trace.profileEvents[0].first->metric), -7);
  ASSERT_EQ(trace.markers.size(), 1);
  EXPECT_EQ(trace.markers[0]->scopeId, 1);
  EXPECT_EQ(trace.markers[0]->cycle, 150);
}

TEST_F(CircularLayoutParserTest, LegacyClockBits) {
  auto append32 = [&](uint32_t word) {
    for (int i = 0; i < 4; ++i)
      testData.push_back((word >> (i * 8)) & 0xff);
  };
  auto append64 = [&](uint64_t word) {
    append32(word);
    append32(word >> 32);
  };

  constexpr uint32_t clockUpper = 0x005a1234;
  append32(kPreamble);
  append32(1);  // block id
  append32(3);  // processor id
  append32(16); // buffer size
  append64(10); // init
  append64(20); // pre-final
  append64(30); // post-final
  append32(4);  // two two-word records
  append32((2u << 23) | clockUpper);
  append32(100);
  append32((1u << 31) | (2u << 23) | clockUpper);
  append32(200);

  config.version = 1;
  config.numBlocks = 1;
  config.totalUnits = 1;
  config.scratchMemSize = testData.size();
  config.uidVec = {0};
  auto buffer = ByteSpan(testData.data(), testData.size());
  CircularLayoutParser parser(buffer, config);
  parser.parse();

  auto &events = parser.getResult()->blockTraces[0].traces[0].profileEvents;
  ASSERT_EQ(events.size(), 1);
  const uint64_t expectedUpper =
      static_cast<uint64_t>(clockUpper) << 32;
  EXPECT_EQ(events[0].first->cycle, expectedUpper | 100);
  EXPECT_EQ(events[0].second->cycle, expectedUpper | 200);
  EXPECT_EQ(events[0].first->eventType, EntryEventType::SCOPE);
  EXPECT_EQ(events[0].first->metricType, EntryMetricType::NONE);
}

TEST_F(CircularLayoutParserTest, StartAfterStart) {
  testData = {
      // header
      0xef, 0xbe, 0xad, 0xde, // preamble
      0x01, 0x00, 0x00, 0x00, // program id
      0x03, 0x00, 0x00, 0x00, // hw id
      0x10, 0x00, 0x00, 0x00, // buf size
      0xef, 0xcd, 0xab, 0x89, // initial time
      0x67, 0x45, 0x23, 0x01, //
      0x10, 0x32, 0x54, 0x76, // pre-final time
      0x98, 0xba, 0xdc, 0xfe, //
      0x08, 0x07, 0x06, 0x05, // post-final time
      0x04, 0x03, 0x02, 0x01, //
      // num events
      0xff, 0x00, 0x00, 0x00,
      // profiled data
      0x04, 0x00, 0x00, 0x00, // start
      0x00, 0x10, 0x00, 0x00, //
      0x04, 0x00, 0x00, 0x00, // start
      0x00, 0x20, 0x00, 0x00, //
  };
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
  testData = {
      // header
      0xef, 0xbe, 0xad, 0xde, // preamble
      0x01, 0x00, 0x00, 0x00, // program id
      0x03, 0x00, 0x00, 0x00, // hw id
      0x30, 0x00, 0x00, 0x00, // buf size
      0xef, 0xcd, 0xab, 0x89, // initial time
      0x67, 0x45, 0x23, 0x01, //
      0x10, 0x32, 0x54, 0x76, // pre-final time
      0x98, 0xba, 0xdc, 0xfe, //
      0x08, 0x07, 0x06, 0x05, // post-final time
      0x04, 0x03, 0x02, 0x01, //
      // num events
      0xff, 0x00, 0x00, 0x00, // segment 0
      0xff, 0x00, 0x00, 0x00, // segment 1
      0xff, 0x00, 0x00, 0x00, // segment 2
      // segment 0
      0x00, 0x00, 0x00, 0x00, // start
      0x00, 0x10, 0x00, 0x00, //
      0x00, 0x00, 0x00, 0x80, // end
      0x00, 0x20, 0x00, 0x00, //
      // segment 1
      0x00, 0x00, 0x00, 0x00, // start
      0x00, 0x10, 0x00, 0x00, //
      0x00, 0x00, 0x00, 0x80, // end
      0x00, 0x20, 0x00, 0x00, //
      // segment 2
      0x00, 0x00, 0x00, 0x00, // start
      0x00, 0x10, 0x00, 0x00, //
      0x00, 0x00, 0x00, 0x80, // end
      0x00, 0x20, 0x00, 0x00, //
      // extra
      0xff, 0xff, 0xff, 0xff, //
      0xff, 0xff, 0xff, 0xff, //
  };
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
  testData = {
      // header
      0xef, 0xbe, 0xad, 0xde, // preamble
      0x01, 0x00, 0x00, 0x00, // program id
      0x03, 0x00, 0x00, 0x00, // hw id
      0x20, 0x00, 0x00, 0x00, // buf size
      0xef, 0xcd, 0xab, 0x89, // initial time
      0x67, 0x45, 0x23, 0x01, //
      0x10, 0x32, 0x54, 0x76, // pre-final time
      0x98, 0xba, 0xdc, 0xfe, //
      0x08, 0x07, 0x06, 0x05, // post-final time
      0x04, 0x03, 0x02, 0x01, //
      // num events
      0xff, 0x00, 0x00, 0x00,
      // profiled data
      0x00, 0x00, 0x00, 0x00, // event 0 start
      0x21, 0x00, 0x00, 0x00, //
      0x00, 0x00, 0x00, 0x01, // event 0 end
      0x36, 0x00, 0x00, 0x00, //
      0x00, 0x00, 0x00, 0x80, // event 1 start
      0x46, 0x00, 0x00, 0x00, //
      0x00, 0x00, 0x00, 0x81, // event 1 end
      0x64, 0x00, 0x00, 0x00, //
  };
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

TEST_F(CircularLayoutParserTest, MarkerTimeShift) {
  auto result = std::make_shared<CircularLayoutParserResult>();
  auto &trace = result->blockTraces.emplace_back().traces.emplace_back();
  auto first = std::make_shared<CycleEntry>();
  first->cycle = 100;
  auto second = std::make_shared<CycleEntry>();
  second->cycle = 120;
  trace.markers = {first, second};

  timeShift(/*cost=*/7, result);

  EXPECT_EQ(first->cycle, 93);
  EXPECT_EQ(second->cycle, 106);
}
