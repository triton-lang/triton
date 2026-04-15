#include "Utility/ProtoWriter.h"

#include <cstring>

namespace proton {

void ProtoWriter::writeUInt64(uint32_t fieldId, uint64_t value) {
  writeTag(fieldId, WireType::Varint);
  writeVarint(value);
}

void ProtoWriter::writeInt64(uint32_t fieldId, int64_t value) {
  writeUInt64(fieldId, static_cast<uint64_t>(value));
}

void ProtoWriter::writeUInt32(uint32_t fieldId, uint32_t value) {
  writeUInt64(fieldId, value);
}

void ProtoWriter::writeInt32(uint32_t fieldId, int32_t value) {
  writeInt64(fieldId, value);
}

void ProtoWriter::writeBool(uint32_t fieldId, bool value) {
  writeUInt64(fieldId, value ? 1 : 0);
}

void ProtoWriter::writeDouble(uint32_t fieldId, double value) {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(value));
  writeTag(fieldId, WireType::Fixed64);
  for (size_t i = 0; i < sizeof(bits); ++i) {
    buffer.push_back(static_cast<char>((bits >> (8 * i)) & 0xff));
  }
}

void ProtoWriter::writeString(uint32_t fieldId, std::string_view value) {
  writeBytes(fieldId, value.data(), value.size());
}

void ProtoWriter::writeMessage(uint32_t fieldId, const ProtoWriter &message) {
  writeBytes(fieldId, message.data().data(), message.data().size());
}

const std::string &ProtoWriter::data() const { return buffer; }

void ProtoWriter::writeTag(uint32_t fieldId, WireType wireType) {
  writeVarint((static_cast<uint64_t>(fieldId) << 3) |
              static_cast<uint8_t>(wireType));
}

void ProtoWriter::writeVarint(uint64_t value) {
  while (value >= 0x80) {
    buffer.push_back(static_cast<char>((value & 0x7f) | 0x80));
    value >>= 7;
  }
  buffer.push_back(static_cast<char>(value));
}

void ProtoWriter::writeBytes(uint32_t fieldId, const char *data,
                             size_t size) {
  writeTag(fieldId, WireType::LengthDelimited);
  writeVarint(static_cast<uint64_t>(size));
  buffer.append(data, size);
}

} // namespace proton
