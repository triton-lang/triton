#ifndef PROTON_UTILITY_STRING_H_
#define PROTON_UTILITY_STRING_H_

#include <string>

namespace proton {

inline std::string toLower(const std::string &str) {
  std::string lower;
  for (auto c : str) {
    lower += tolower(c);
  }
  return lower;
}

} // namespace proton

#endif // PROTON_UTILITY_STRING_H_
