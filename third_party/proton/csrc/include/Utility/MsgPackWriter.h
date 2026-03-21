#ifndef PROTON_UTILITY_MSGPACK_WRITER_H_
#define PROTON_UTILITY_MSGPACK_WRITER_H_

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace proton {

// See https://msgpack.org/index.html for the specification.
class MsgPackWriter {
public:
  void reserve(size_t bytes);

  std::vector<uint8_t> take() &&;

  void packNil();
  void packBool(bool value);
  void packUInt(uint64_t value);
  void packInt(int64_t value);
  void packDouble(double value);
  void packStr(std::string_view value);
  void packArray(uint32_t size);
  void packMap(uint32_t size);

private:
  std::vector<uint8_t> out;
};

} // namespace proton

#endif // PROTON_UTILITY_MSGPACK_WRITER_H_
