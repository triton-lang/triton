#ifndef PROTON_UTILITY_TABLE_H_
#define PROTON_UTILITY_TABLE_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace proton {

// Dense table for ids in a contiguous range [minId, maxId].
template <typename T, typename IdT = uint64_t>
class RangeTable {
  static_assert(std::is_integral_v<IdT>, "RangeTable IdT must be integral");

public:
  void resetRange(IdT minId, IdT maxId) {
    if (maxId < minId) {
      clear();
      return;
    }
    minId_ = minId;
    auto size = static_cast<size_t>(maxId - minId + 1);
    nodes_.clear();
    nodes_.resize(size);
    present_.assign(size, 0);
  }

  void clear() {
    minId_ = 0;
    nodes_.clear();
    present_.clear();
  }

  std::pair<T *, bool> tryEmplace(IdT id) {
    if (!inRange(id))
      return {nullptr, false};
    auto index = indexFor(id);
    bool inserted = !present_[index];
    present_[index] = 1;
    return {&nodes_[index], inserted};
  }

  T *find(IdT id) {
    if (!inRange(id))
      return nullptr;
    auto index = indexFor(id);
    return present_[index] ? &nodes_[index] : nullptr;
  }

  const T *find(IdT id) const {
    if (!inRange(id))
      return nullptr;
    auto index = indexFor(id);
    return present_[index] ? &nodes_[index] : nullptr;
  }

  bool empty() const { return nodes_.empty(); }

private:
  bool inRange(IdT id) const {
    if (nodes_.empty() || id < minId_)
      return false;
    auto offset = static_cast<size_t>(id - minId_);
    return offset < nodes_.size();
  }

  size_t indexFor(IdT id) const {
    return static_cast<size_t>(id - minId_);
  }

  IdT minId_{0};
  std::vector<T> nodes_;
  std::vector<uint8_t> present_;
};

} // namespace proton

#endif // PROTON_UTILITY_TABLE_H_
