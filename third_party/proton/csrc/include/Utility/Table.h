#ifndef PROTON_UTILITY_TABLE_H_
#define PROTON_UTILITY_TABLE_H_

#include <algorithm>
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
    ensureRange(minIdValue, maxIdValue);
    clear();
  }

  void clear() {
    std::fill(present.begin(), present.end(), false);
    liveCount = 0;
  }

  std::pair<T *, bool> tryEmplace(IdT id) {
    ensureRange(id, id);
    auto index = indexFor(id);
    bool inserted = !present[index];
    if (inserted) {
      present[index] = true;
      ++liveCount;
      nodes[index] = T{};
    }
    return {&nodes[index], inserted};
  }

  T &emplace(IdT id) {
    return *tryEmplace(id).first;
  }

  void erase(IdT id) {
    if (!inRange(id))
      return;
    auto index = indexFor(id);
    if (!present[index])
      return;
    present[index] = false;
    nodes[index] = T{};
    --liveCount;
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

  bool empty() const { return liveCount == 0; }

  size_t size() const { return liveCount; }

private:
  void ensureRange(IdT minIdValue, IdT maxIdValue) {
    if (nodes.empty()) {
      minId = minIdValue;
      auto size = static_cast<size_t>(maxIdValue - minIdValue + 1);
      nodes.resize(size);
      present.assign(size, false);
      return;
    }

    if (minIdValue < minId) {
      auto prefix = static_cast<size_t>(minId - minIdValue);
      nodes.insert(nodes.begin(), prefix, T{});
      present.insert(present.begin(), prefix, false);
      minId = minIdValue;
    }

    auto maxId = minId + static_cast<IdT>(nodes.size() - 1);
    if (maxIdValue > maxId) {
      auto suffix = static_cast<size_t>(maxIdValue - maxId);
      nodes.resize(nodes.size() + suffix);
      present.resize(present.size() + suffix, false);
    }
  }

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
  size_t liveCount{0};
};

} // namespace proton

#endif // PROTON_UTILITY_TABLE_H_
