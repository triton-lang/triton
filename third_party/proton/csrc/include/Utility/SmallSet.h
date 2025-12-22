#ifndef PROTON_UTILITY_SMALL_SET_H_
#define PROTON_UTILITY_SMALL_SET_H_

#include <algorithm>
#include <cstddef>
#include <vector>

namespace proton {

/// A simple small set optimized for tiny cardinalities with linear lookup.
template <typename Key, size_t InlineReserve = 4> class SmallSet {
public:
  SmallSet() { data.reserve(InlineReserve); }

  bool insert(const Key &key) {
    if (contains(key))
      return false;
    data.push_back(key);
    return true;
  }

  bool erase(const Key &key) {
    auto it = std::find(data.begin(), data.end(), key);
    if (it == data.end())
      return false;
    *it = data.back();
    data.pop_back();
    return true;
  }

  bool contains(const Key &key) const {
    return std::find(data.begin(), data.end(), key) != data.end();
  }

  void clear() { data.clear(); }

  size_t size() const { return data.size(); }

  bool empty() const { return data.empty(); }

  auto begin() { return data.begin(); }
  auto end() { return data.end(); }
  auto begin() const { return data.begin(); }
  auto end() const { return data.end(); }

private:
  std::vector<Key> data;
};

} // namespace proton

#endif // PROTON_UTILITY_SMALL_SET_H_
