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

inline std::string replace(const std::string &str, const std::string &src,
                           const std::string &dst) {
  std::string replaced = str;
  size_t pos = 0;
  while ((pos = replaced.find(src, pos)) != std::string::npos) {
    replaced.replace(pos, src.length(), dst);
    pos += dst.length();
  }
  return replaced;
}

} // namespace proton

#endif // PROTON_UTILITY_STRING_H_
