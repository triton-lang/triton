#include "TraceDataIO/EntryDecoder.h"
#include <cstring>

using namespace proton;

std::ostream &operator<<(std::ostream &os, const EntryBase &obj) {
  obj.print(os);
  return os;
}

void I32Entry::print(std::ostream &os) const { os << value; }

template <> void proton::decodeFn<I32Entry>(ByteSpan &buffer, I32Entry &entry) {
  entry.value = buffer.readInt32();
}

void I64Entry::print(std::ostream &os) const { os << value; }

template <> void proton::decodeFn<I64Entry>(ByteSpan &buffer, I64Entry &entry) {
  entry.value = buffer.readInt64();
}

void CycleEntry::print(std::ostream &os) const {
  std::string prefix = isStart ? "S" : "E";
  os << prefix + std::to_string(scopeId) + "C" + std::to_string(cycle);
}

template <>
void proton::decodeFn<CycleEntry>(ByteSpan &buffer, CycleEntry &entry) {
  uint32_t tagClkUpper = buffer.readUInt32();
  entry.rawTag = tagClkUpper;
  entry.isStart = (tagClkUpper & 0x80000000) == 0;
  entry.scopeId = (tagClkUpper & 0x7F800000) >> 23;
  entry.eventType =
      static_cast<EntryEventType>((tagClkUpper & 0x00700000) >> 20);
  entry.metricType =
      static_cast<EntryMetricType>((tagClkUpper & 0x000F0000) >> 16);
  entry.rawValue = buffer.readUInt32();
  entry.cycle =
      static_cast<uint64_t>(tagClkUpper & 0x7FF) << 32 | entry.rawValue;
}

namespace {

float bitsToFloat(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

double fp16ToDouble(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000) << 16;
  uint32_t exponent = (bits >> 10) & 0x1f;
  uint32_t fraction = bits & 0x03ff;
  uint32_t fp32Bits;
  if (exponent == 0) {
    if (fraction == 0) {
      fp32Bits = sign;
    } else {
      int shift = 0;
      while ((fraction & 0x0400) == 0) {
        fraction <<= 1;
        ++shift;
      }
      fraction &= 0x03ff;
      fp32Bits = sign | static_cast<uint32_t>(127 - 14 - shift) << 23 |
                 fraction << 13;
    }
  } else if (exponent == 0x1f) {
    fp32Bits = sign | 0x7f800000 | fraction << 13;
  } else {
    fp32Bits = sign | (exponent + (127 - 15)) << 23 | fraction << 13;
  }
  return static_cast<double>(bitsToFloat(fp32Bits));
}

} // namespace

EntryMetricValue proton::decodeMetricValue(EntryMetricType type,
                                           uint32_t rawValue) {
  switch (type) {
  case EntryMetricType::BOOL:
    return static_cast<uint64_t>(rawValue != 0);
  case EntryMetricType::I8:
    return static_cast<int64_t>(static_cast<int8_t>(rawValue));
  case EntryMetricType::I16:
    return static_cast<int64_t>(static_cast<int16_t>(rawValue));
  case EntryMetricType::I32:
    return static_cast<int64_t>(static_cast<int32_t>(rawValue));
  case EntryMetricType::U8:
    return static_cast<uint64_t>(static_cast<uint8_t>(rawValue));
  case EntryMetricType::U16:
    return static_cast<uint64_t>(static_cast<uint16_t>(rawValue));
  case EntryMetricType::U32:
    return static_cast<uint64_t>(rawValue);
  case EntryMetricType::F16:
    return fp16ToDouble(static_cast<uint16_t>(rawValue));
  case EntryMetricType::BF16:
    return static_cast<double>(bitsToFloat(rawValue << 16));
  case EntryMetricType::F32:
    return static_cast<double>(bitsToFloat(rawValue));
  case EntryMetricType::NONE:
    break;
  }
  throw makeInvalidArgument("Cannot decode an in-kernel metric without a type");
}
