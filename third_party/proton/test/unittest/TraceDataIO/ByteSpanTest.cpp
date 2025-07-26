#include "TraceDataIO/ByteSpan.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>

using namespace proton;

TEST(ByteSpanTest, ReadAndNavigation) {
  std::vector<uint8_t> testData = {
      // int8 values (positions 0-3)
      0x00, // 0
      0x7F, // 127
      0x80, // -128
      0xFF, // -1

      // int16 values (positions 4-7)
      0x34, 0x12, // 0x1234
      0x00, 0x80, // 0x8000

      // int32 values (positions 8-15)
      0x78, 0x56, 0x34, 0x12, // 0x12345678
      0x00, 0x00, 0x00, 0x80  // 0x80000000
  };

  ByteSpan span(testData.data(), testData.size());

  // Test initial state
  EXPECT_EQ(span.position(), 0);
  EXPECT_EQ(span.size(), 16);
  EXPECT_EQ(span.remaining(), 16);
  EXPECT_TRUE(span.hasRemaining(16));
  EXPECT_FALSE(span.hasRemaining(17));

  // Test 8-bit reading
  EXPECT_EQ(span.readInt8(), 0);
  EXPECT_EQ(span.readInt8(), 127);
  EXPECT_EQ(span.readUInt8(), 128);
  EXPECT_EQ(span.readUInt8(), 255);
  EXPECT_EQ(span.position(), 4);

  // Test navigation - seeking back
  span.seek(1);
  EXPECT_EQ(span.position(), 1);
  EXPECT_EQ(span.readInt8(), 127);
  EXPECT_EQ(span.position(), 2);

  // Test navigation - skipping
  span.skip(2);
  EXPECT_EQ(span.position(), 4);

  // Test 16-bit reading
  EXPECT_EQ(span.readUInt16(), 0x1234); // 0x1234
  EXPECT_EQ(span.readInt16(), -32768);  // 0x8000
  EXPECT_EQ(span.position(), 8);

  // Test navigation - seeking to specific position
  span.seek(8);

  // Test 32-bit reading
  EXPECT_EQ(span.readUInt32(), 305419896);  // 0x12345678
  EXPECT_EQ(span.readInt32(), -2147483648); // 0x80000000
  EXPECT_EQ(span.position(), 16);

  // Test navigation - buffer overflow
  EXPECT_THROW(span.skip(1), BufferException);
  EXPECT_THROW(span.seek(17), BufferException);

  // Test navigation - at the end
  EXPECT_EQ(span.remaining(), 0);
  EXPECT_FALSE(span.hasRemaining(1));
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
