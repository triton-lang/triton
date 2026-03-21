#include "Utility/MsgPackWriter.h"

#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

namespace proton {
namespace {

template <typename T> void writeBE(std::vector<uint8_t> &out, T value) {
  using U = std::make_unsigned_t<T>;
  U u = static_cast<U>(value);
  for (int i = sizeof(U) - 1; i >= 0; --i) {
    out.push_back(static_cast<uint8_t>((u >> (i * 8)) & 0xff));
  }
}

} // namespace

void MsgPackWriter::reserve(size_t bytes) { out.reserve(bytes); }

std::vector<uint8_t> MsgPackWriter::take() && { return std::move(out); }

void MsgPackWriter::packNil() { out.push_back(0xc0); }

void MsgPackWriter::packBool(bool value) { out.push_back(value ? 0xc3 : 0xc2); }

void MsgPackWriter::packUInt(uint64_t value) {
  if (value <= 0x7f) {
    out.push_back(static_cast<uint8_t>(value));
  } else if (value <= 0xff) {
    out.push_back(0xcc);
    out.push_back(static_cast<uint8_t>(value));
  } else if (value <= 0xffff) {
    out.push_back(0xcd);
    writeBE(out, static_cast<uint16_t>(value));
  } else if (value <= 0xffffffffull) {
    out.push_back(0xce);
    writeBE(out, static_cast<uint32_t>(value));
  } else {
    out.push_back(0xcf);
    writeBE(out, static_cast<uint64_t>(value));
  }
}

void MsgPackWriter::packInt(int64_t value) {
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
    writeBE(out, static_cast<int16_t>(value));
  } else if (value >= std::numeric_limits<int32_t>::min()) {
    out.push_back(0xd2);
    writeBE(out, static_cast<int32_t>(value));
  } else {
    out.push_back(0xd3);
    writeBE(out, static_cast<int64_t>(value));
  }
}

void MsgPackWriter::packDouble(double value) {
  out.push_back(0xcb);
  uint64_t bits{};
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  writeBE(out, bits);
}

void MsgPackWriter::packStr(std::string_view value) {
  const auto size = static_cast<uint32_t>(value.size());
  if (size <= 31) {
    out.push_back(static_cast<uint8_t>(0xa0 | size));
  } else if (size <= 0xff) {
    out.push_back(0xd9);
    out.push_back(static_cast<uint8_t>(size));
  } else if (size <= 0xffff) {
    out.push_back(0xda);
    writeBE(out, static_cast<uint16_t>(size));
  } else {
    out.push_back(0xdb);
    writeBE(out, static_cast<uint32_t>(size));
  }
  out.insert(out.end(), value.begin(), value.end());
}

void MsgPackWriter::packArray(uint32_t size) {
  if (size <= 15) {
    out.push_back(static_cast<uint8_t>(0x90 | size));
  } else if (size <= 0xffff) {
    out.push_back(0xdc);
    writeBE(out, static_cast<uint16_t>(size));
  } else {
    out.push_back(0xdd);
    writeBE(out, static_cast<uint32_t>(size));
  }
}

void MsgPackWriter::packMap(uint32_t size) {
  if (size <= 15) {
    out.push_back(static_cast<uint8_t>(0x80 | size));
  } else if (size <= 0xffff) {
    out.push_back(0xde);
    writeBE(out, static_cast<uint16_t>(size));
  } else {
    out.push_back(0xdf);
    writeBE(out, static_cast<uint32_t>(size));
  }
}

} // namespace proton
