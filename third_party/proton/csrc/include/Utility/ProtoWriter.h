#ifndef PROTON_UTILITY_PROTO_WRITER_H_
#define PROTON_UTILITY_PROTO_WRITER_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace proton {

// Minimal protobuf wire-format writer for the Perfetto trace export path.
class ProtoWriter {
public:
  void writeUInt64(uint32_t fieldId, uint64_t value);
  void writeInt64(uint32_t fieldId, int64_t value);
  void writeUInt32(uint32_t fieldId, uint32_t value);
  void writeInt32(uint32_t fieldId, int32_t value);
  void writeBool(uint32_t fieldId, bool value);
  void writeDouble(uint32_t fieldId, double value);
  void writeString(uint32_t fieldId, std::string_view value);
  void writeMessage(uint32_t fieldId, const ProtoWriter &message);

  const std::string &data() const;

private:
  enum class WireType : uint8_t {
    Varint = 0,
    Fixed64 = 1,
    LengthDelimited = 2
  };

  void writeTag(uint32_t fieldId, WireType wireType);
  void writeVarint(uint64_t value);
  void writeBytes(uint32_t fieldId, const char *data, size_t size);

  std::string buffer;
};

} // namespace proton

#endif // PROTON_UTILITY_PROTO_WRITER_H_
