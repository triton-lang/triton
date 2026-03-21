#ifndef PROTON_UTILITY_TABLE_H_
#define PROTON_UTILITY_TABLE_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace proton {

// Dense table for ids in a contiguous range [minId, maxId].
template <typename T, typename IdT = uint64_t> class RangeTable {
  static_assert(std::is_integral_v<IdT>, "RangeTable IdT must be integral");

public:
  void resetRange(IdT minIdValue, IdT maxIdValue) {
    if (maxIdValue < minIdValue) {
      clear();
      return;
    }
    minId = minIdValue;
    auto size = static_cast<size_t>(maxIdValue - minIdValue + 1);
    nodes.clear();
    nodes.resize(size);
    present.assign(size, false);
  }

  void clear() {
    minId = 0;
    nodes.clear();
    present.clear();
  }

  std::pair<T *, bool> tryEmplace(IdT id) {
    if (!inRange(id))
      return {nullptr, false};
    auto index = indexFor(id);
    bool inserted = !present[index];
    present[index] = true;
    return {&nodes[index], inserted};
  }

  T &emplace(IdT id) {
    auto index = indexFor(id);
    present[index] = true;
    return nodes[index];
  }

  T *find(IdT id) {
    if (!inRange(id))
      return nullptr;
    auto index = indexFor(id);
    return present[index] ? &nodes[index] : nullptr;
  }

  const T *find(IdT id) const {
    if (!inRange(id))
      return nullptr;
    auto index = indexFor(id);
    return present[index] ? &nodes[index] : nullptr;
  }

  bool empty() const { return nodes.empty(); }

private:
  bool inRange(IdT id) const {
    if (nodes.empty() || id < minId)
      return false;
    auto offset = static_cast<size_t>(id - minId);
    return offset < nodes.size();
  }

  size_t indexFor(IdT id) const { return static_cast<size_t>(id - minId); }

  IdT minId{0};
  std::vector<T> nodes;
  std::vector<bool> present;
};

} // namespace proton

#endif // PROTON_UTILITY_TABLE_H_
