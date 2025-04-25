#include "Data/TraceDataIO/ByteSpan.h"

using namespace proton;

ByteSpan::ByteSpan(const uint8_t *data, size_t size)
    : data_(data), size_(size), position_(0) {
  if (data == nullptr && size > 0) {
    throw std::invalid_argument(
        "Data pointer cannot be null for non-zero size");
  }
}

void ByteSpan::checkRemaining(size_t required) const {
  if (remaining() < required) {
    throw BufferException("");
  }
}

uint8_t ByteSpan::readUInt8() {
  checkRemaining(1);
  return data_[position_++];
}

int8_t ByteSpan::readInt8() { return static_cast<int8_t>(readUInt8()); }

uint16_t ByteSpan::readUInt16() {
  checkRemaining(2);
  uint16_t value = static_cast<uint16_t>(data_[position_]) |
                   (static_cast<uint16_t>(data_[position_ + 1]) << 8);
  position_ += 2;
  return value;
}

int16_t ByteSpan::readInt16() { return static_cast<int16_t>(readUInt16()); }

uint32_t ByteSpan::readUInt32() {
  checkRemaining(4);
  uint32_t value = static_cast<uint32_t>(data_[position_]) |
                   (static_cast<uint32_t>(data_[position_ + 1]) << 8) |
                   (static_cast<uint32_t>(data_[position_ + 2]) << 16) |
                   (static_cast<uint32_t>(data_[position_ + 3]) << 24);
  position_ += 4;
  return value;
}

int32_t ByteSpan::readInt32() { return static_cast<int32_t>(readUInt32()); }

void ByteSpan::skip(size_t count) {
  checkRemaining(count);
  position_ += count;
}

void ByteSpan::seek(size_t position) {
  if (position > size_) {
    throw BufferException("");
  }
  position_ = position;
}

BufferException::BufferException(const std::string &message)
    : std::runtime_error(message) {}
