#ifndef TRITON_TOOLS_STLEXTRAS_H
#define TRITON_TOOLS_STLEXTRAS_H

#include <tuple>

namespace mlir::triton {
// Iterate over a parameter pack with an index.
template <typename F, typename... Args>
auto for_each_with_index(F f, Args &&...args) {
  size_t index = 0;
  return std::make_tuple(([&]() {
    return f(index++, std::forward<Args>(args));
  }())...);
}
} // namespace mlir::triton

#endif // TRITON_TOOLS_STLEXTRAS_H
