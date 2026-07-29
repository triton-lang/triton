#ifndef PROTON_COMMON_ENTRY_DECODER_H_
#define PROTON_COMMON_ENTRY_DECODER_H_

#include "ByteSpan.h"
#include "Utility/Errors.h"
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace proton {

class EntryBase;

template <typename EntryT> void decodeFn(ByteSpan &buffer, EntryT &entry) {
  throw makeLogicError("No decoder function is implemented");
}

class EntryDecoder {
private:
  ByteSpan &buf;

public:
  explicit EntryDecoder(ByteSpan &buffer) : buf(buffer) {}

  template <typename EntryT> std::shared_ptr<EntryT> decode() {
    auto entry = std::make_shared<EntryT>();
    decodeFn<EntryT>(buffer(), *entry);
    return entry;
  }

protected:
  // Protected accessor for the buffer
  ByteSpan &buffer() { return buf; }
};

struct EntryBase {
  virtual ~EntryBase() = default;

  virtual void print(std::ostream &os) const = 0;
};

std::ostream &operator<<(std::ostream &os, const EntryBase &obj);

struct I32Entry : public EntryBase {
  I32Entry() = default;

  void print(std::ostream &os) const override;

  int32_t value = 0;
};

template <> void decodeFn<I32Entry>(ByteSpan &buffer, I32Entry &entry);

struct I64Entry : public EntryBase {
  I64Entry() = default;

  void print(std::ostream &os) const override;

  int64_t value = 0;
};

template <> void decodeFn<I64Entry>(ByteSpan &buffer, I64Entry &entry);

enum class EntryEventType : uint8_t {
  SCOPE = 0,
  ASYNC = 1,
  MARK = 2,
  METRIC = 3,
};

enum class EntryMetricType : uint8_t {
  NONE = 0,
  BOOL = 1,
  I8 = 2,
  I16 = 3,
  I32 = 4,
  U8 = 5,
  U16 = 6,
  U32 = 7,
  F16 = 8,
  BF16 = 9,
  F32 = 10,
};

using EntryMetricValue = std::variant<uint64_t, int64_t, double>;

struct CycleEntry : public EntryBase {
  CycleEntry() = default;

  void print(std::ostream &os) const override;

  uint64_t cycle = 0;
  bool isStart = true;
  int32_t scopeId = 0;
  EntryEventType eventType = EntryEventType::SCOPE;
  EntryMetricType metricType = EntryMetricType::NONE;
  uint32_t rawValue = 0;
  std::optional<EntryMetricValue> metric;
  std::optional<std::string> metricName;
};

template <> void decodeFn<CycleEntry>(ByteSpan &buffer, CycleEntry &entry);

EntryMetricValue decodeMetricValue(EntryMetricType type, uint32_t rawValue);

} // namespace proton

#endif // PROTON_COMMON_ENTRY_DECODER_H_
