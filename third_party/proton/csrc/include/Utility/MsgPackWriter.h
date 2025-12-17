#ifndef PROTON_UTILITY_MSGPACK_WRITER_H_
#define PROTON_UTILITY_MSGPACK_WRITER_H_

#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

namespace proton {

class MsgPackWriter {
public:
  void reserve(size_t bytes) { out.reserve(bytes); }

  std::vector<uint8_t> take() && { return std::move(out); }

  void packNil() { out.push_back(0xc0); }

  void packBool(bool value) { out.push_back(value ? 0xc3 : 0xc2); }

  void packUInt(uint64_t value) {
    if (value <= 0x7f) {
      out.push_back(static_cast<uint8_t>(value));
    } else if (value <= 0xff) {
      out.push_back(0xcc);
      out.push_back(static_cast<uint8_t>(value));
    } else if (value <= 0xffff) {
      out.push_back(0xcd);
      writeBE(static_cast<uint16_t>(value));
    } else if (value <= 0xffffffffull) {
      out.push_back(0xce);
      writeBE(static_cast<uint32_t>(value));
    } else {
      out.push_back(0xcf);
      writeBE(static_cast<uint64_t>(value));
    }
  }

  void packInt(int64_t value) {
    if (value >= 0) {
      packUInt(static_cast<uint64_t>(value));
      return;
    }
    if (value >= -32) {
      out.push_back(static_cast<uint8_t>(0xe0 | (value + 32)));
    } else if (value >= std::numeric_limits<int8_t>::min()) {
      out.push_back(0xd0);
      out.push_back(static_cast<uint8_t>(static_cast<int8_t>(value)));
    } else if (value >= std::numeric_limits<int16_t>::min()) {
      out.push_back(0xd1);
      writeBE(static_cast<int16_t>(value));
    } else if (value >= std::numeric_limits<int32_t>::min()) {
      out.push_back(0xd2);
      writeBE(static_cast<int32_t>(value));
    } else {
      out.push_back(0xd3);
      writeBE(static_cast<int64_t>(value));
    }
  }

  void packDouble(double value) {
    out.push_back(0xcb);
    uint64_t bits{};
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    writeBE(bits);
  }

  void packStr(std::string_view value) {
    const auto size = static_cast<uint32_t>(value.size());
    if (size <= 31) {
      out.push_back(static_cast<uint8_t>(0xa0 | size));
    } else if (size <= 0xff) {
      out.push_back(0xd9);
      out.push_back(static_cast<uint8_t>(size));
    } else if (size <= 0xffff) {
      out.push_back(0xda);
      writeBE(static_cast<uint16_t>(size));
    } else {
      out.push_back(0xdb);
      writeBE(static_cast<uint32_t>(size));
    }
    out.insert(out.end(), value.begin(), value.end());
  }

  void packArray(uint32_t size) {
    if (size <= 15) {
      out.push_back(static_cast<uint8_t>(0x90 | size));
    } else if (size <= 0xffff) {
      out.push_back(0xdc);
      writeBE(static_cast<uint16_t>(size));
    } else {
      out.push_back(0xdd);
      writeBE(static_cast<uint32_t>(size));
    }
  }

  void packMap(uint32_t size) {
    if (size <= 15) {
      out.push_back(static_cast<uint8_t>(0x80 | size));
    } else if (size <= 0xffff) {
      out.push_back(0xde);
      writeBE(static_cast<uint16_t>(size));
    } else {
      out.push_back(0xdf);
      writeBE(static_cast<uint32_t>(size));
    }
  }

private:
  template <typename T> void writeBE(T value) {
    using U = std::make_unsigned_t<T>;
    U u = static_cast<U>(value);
    for (int i = sizeof(U) - 1; i >= 0; --i) {
      out.push_back(static_cast<uint8_t>((u >> (i * 8)) & 0xff));
    }
  }

  std::vector<uint8_t> out;
};

} // namespace proton

#endif // PROTON_UTILITY_MSGPACK_WRITER_H_
