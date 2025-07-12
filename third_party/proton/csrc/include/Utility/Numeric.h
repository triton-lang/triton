#ifndef PROTON_UTILITY_NUMERIC_H_
#define PROTON_UTILITY_NUMERIC_H_

#include <cstddef>

namespace proton {

template <typename T> constexpr T nextPowerOfTwo(T value) {
  if (value < 1) {
    return 1;
  }
  --value; // Decrement to handle the case where value is already a power of two
  for (size_t i = 1; i < sizeof(T) * 8; i <<= 1) {
    value |= value >> i; // Propagate the highest set bit to the right
  }
  return value + 1; // Increment to get the next power of two
}

} // namespace proton

#endif // PROTON_UTILITY_NUMERIC_H_
