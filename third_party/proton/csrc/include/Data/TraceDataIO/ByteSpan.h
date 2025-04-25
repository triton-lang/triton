#ifndef PROTON_DATA_BYTE_SPAN_H_
#define PROTON_DATA_BYTE_SPAN_H_

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
  size_t position() const { return position_; }
  size_t size() const { return size_; }
  size_t remaining() const { return size_ - position_; }
  bool hasRemaining(size_t count = 1) const { return remaining() >= count; }

  // Data access
  const uint8_t *data() const { return data_; }
  const uint8_t *currentData() const { return data_ + position_; }

private:
  const uint8_t *data_; // Pointer to the underlying data
  size_t size_;         // Total size of the data
  size_t position_;     // Current read position

  // Helper method to check remaining bytes
  void checkRemaining(size_t required) const;
};

} // namespace proton

#endif // PROTON_DATA_BYTE_SPAN_H_
