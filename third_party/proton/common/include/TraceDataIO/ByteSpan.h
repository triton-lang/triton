#ifndef PROTON_COMMON_BYTE_SPAN_H_
#define PROTON_COMMON_BYTE_SPAN_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace proton {

class BufferException : public std::runtime_error {
public:
  explicit BufferException(const std::string &message);
};

class ByteSpan {
public:
  ByteSpan(const uint8_t *data, size_t size);

  // Read methods
  uint8_t readUInt8();
  int8_t readInt8();
  uint16_t readUInt16();
  int16_t readInt16();
  uint32_t readUInt32();
  int32_t readInt32();

  // Buffer navigation
  void skip(size_t count);
  void seek(size_t position);
  size_t position() const { return pos; }
  size_t size() const { return dataSize; }
  size_t remaining() const { return dataSize - pos; }
  bool hasRemaining(size_t count = 0) const { return remaining() >= count; }

  // Data access
  const uint8_t *data() const { return dataPtr; }
  const uint8_t *currentData() const { return dataPtr + pos; }

private:
  const uint8_t *dataPtr; // Pointer to the underlying data
  size_t dataSize;        // Total size of the data
  size_t pos;             // Current read position

  // Helper method to check remaining bytes
  void checkRemaining(size_t required) const;
};

} // namespace proton

#endif // PROTON_COMMON_BYTE_SPAN_H_
