#include "TraceDataIO/EntryDecoder.h"

using namespace proton;

std::ostream &operator<<(std::ostream &os, const EntryBase &obj) {
  obj.print(os);
  return os;
}

void I32Entry::print(std::ostream &os) const { os << value; }

template <> void proton::decodeFn<I32Entry>(ByteSpan &buffer, I32Entry &entry) {
  entry.value = buffer.readInt32();
}

void CycleEntry::print(std::ostream &os) const {
  std::string prefix = isStart ? "S" : "E";
  os << prefix + std::to_string(scopeId) + "C" + std::to_string(cycle);
}

template <>
void proton::decodeFn<CycleEntry>(ByteSpan &buffer, CycleEntry &entry) {
  uint32_t tag = buffer.readUInt32();
  entry.isStart = (tag & 0x80000000) == 0;
  entry.scopeId = (tag & 0x7FFFFFFF);
  entry.cycle = buffer.readUInt32();
}
