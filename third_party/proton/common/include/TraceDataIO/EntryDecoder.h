#ifndef PROTON_COMMON_ENTRY_DECODER_H_
#define PROTON_COMMON_ENTRY_DECODER_H_

#include "ByteSpan.h"
#include <cstdint>
#include <iostream>
#include <memory>

namespace proton {

class EntryBase;

template <typename EntryT> void decodeFn(ByteSpan &buffer, EntryT &entry) {
  throw std::runtime_error("No decoder function is implemented");
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

struct CycleEntry : public EntryBase {
  CycleEntry() = default;

  void print(std::ostream &os) const override;

  uint64_t cycle = 0;
  bool isStart = true;
  int32_t scopeId = 0;
};

template <> void decodeFn<CycleEntry>(ByteSpan &buffer, CycleEntry &entry);

} // namespace proton

#endif // PROTON_COMMON_ENTRY_DECODER_H_
