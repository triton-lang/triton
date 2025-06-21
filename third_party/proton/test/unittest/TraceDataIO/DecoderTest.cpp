#include "TraceDataIO/EntryDecoder.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
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
