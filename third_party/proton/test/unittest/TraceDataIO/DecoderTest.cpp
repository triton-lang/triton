#include "TraceDataIO/EntryDecoder.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>

using namespace proton;

TEST(DecoderTest, Decode) {
  std::vector<uint8_t> testData = {0x78, 0x56, 0x34, 0x12, 0x01, 0x00,
                                   0x00, 0x80, 0xFF, 0xFF, 0xFF, 0xFF};

  auto buf = ByteSpan(testData.data(), testData.size());
  auto decoder = EntryDecoder(buf);
  auto entry1 = decoder.decode<I32Entry>();
  EXPECT_EQ(entry1->value, 0x12345678);
  auto entry2 = decoder.decode<CycleEntry>();
  EXPECT_EQ(entry2->isStart, false);
  EXPECT_EQ(entry2->scopeId, 0);
  EXPECT_EQ(entry2->cycle, 8589934591);
}

TEST(DecoderTest, DecodeMetricExtension) {
  const uint32_t tag = (2u << 23) | (3u << 20) | (4u << 16);
  const uint32_t value = static_cast<uint32_t>(-7);
  std::vector<uint8_t> testData;
  auto append = [&](uint32_t word) {
    for (int i = 0; i < 4; ++i)
      testData.push_back((word >> (i * 8)) & 0xff);
  };
  append(tag);
  append(value);

  auto buffer = ByteSpan(testData.data(), testData.size());
  auto entry = EntryDecoder(buffer).decode<CycleEntry>();
  EXPECT_EQ(entry->eventType, EntryEventType::METRIC);
  EXPECT_EQ(entry->metricType, EntryMetricType::I32);
  EXPECT_EQ(entry->scopeId, 2);
  EXPECT_EQ(std::get<int64_t>(
                decodeMetricValue(entry->metricType, entry->rawValue)),
            -7);
}

TEST(DecoderTest, DecodeMetricValues) {
  EXPECT_EQ(std::get<uint64_t>(
                decodeMetricValue(EntryMetricType::BOOL, 0xff)),
            1);
  EXPECT_EQ(
      std::get<int64_t>(decodeMetricValue(EntryMetricType::I8, 0xff)), -1);
  EXPECT_EQ(
      std::get<int64_t>(decodeMetricValue(EntryMetricType::I16, 0xffff)), -1);
  EXPECT_EQ(std::get<int64_t>(
                decodeMetricValue(EntryMetricType::I32, 0xffffffff)),
            -1);
  EXPECT_EQ(
      std::get<uint64_t>(decodeMetricValue(EntryMetricType::U8, 0x1ff)),
      0xff);
  EXPECT_EQ(
      std::get<uint64_t>(decodeMetricValue(EntryMetricType::U16, 0x1ffff)),
      0xffff);
  EXPECT_EQ(std::get<uint64_t>(
                decodeMetricValue(EntryMetricType::U32, 0xffffffff)),
            0xffffffff);
  EXPECT_DOUBLE_EQ(
      std::get<double>(decodeMetricValue(EntryMetricType::F16, 0x3e00)),
      1.5);
  EXPECT_DOUBLE_EQ(
      std::get<double>(decodeMetricValue(EntryMetricType::F16, 0x0001)),
      0x1p-24);
  EXPECT_DOUBLE_EQ(
      std::get<double>(decodeMetricValue(EntryMetricType::BF16, 0x3fc0)),
      1.5);
  EXPECT_DOUBLE_EQ(std::get<double>(
                       decodeMetricValue(EntryMetricType::F32, 0x3fc00000)),
                   1.5);
  EXPECT_THROW(decodeMetricValue(EntryMetricType::NONE, 0),
               std::invalid_argument);
}
