#ifndef PROTON_UTILITY_MSGPACK_WRITER_H_
#define PROTON_UTILITY_MSGPACK_WRITER_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

namespace proton {

// See https://msgpack.org/index.html for the specification.
class MsgPackWriter {
public:
  void reserve(size_t bytes);
  size_t size() const { return out.size(); }
  const uint8_t *data() const { return out.data(); }

  std::vector<uint8_t> take() &&;
  void appendBytes(const std::vector<uint8_t> &bytes);

  void packNil();
  void packBool(bool value);
  void packUInt(uint64_t value);
  void packInt(int64_t value);
  void packDouble(double value);
  void packStr(std::string_view value);
  template <size_t N> void packFixStrLiteral(const char (&value)[N]) {
    static_assert(N > 0);
    constexpr uint32_t size = static_cast<uint32_t>(N - 1);
    // MsgPack fixstr stores the string length in 5 bits, so literals must fit
    // in the 0..31 byte range.
    static_assert(size <= 31);
    out.push_back(static_cast<uint8_t>(0xa0 | size));
    const auto offset = out.size();
    out.resize(offset + size);
    std::memcpy(out.data() + offset, value, size);
  }
  void packArray(uint32_t size);
  void packMap(uint32_t size);

private:
  std::vector<uint8_t> out;
};

} // namespace proton

#endif // PROTON_UTILITY_MSGPACK_WRITER_H_
