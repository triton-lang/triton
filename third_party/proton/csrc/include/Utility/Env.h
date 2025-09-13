#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>

static std::mutex getenv_mutex;

inline bool getBoolEnv(const std::string &env) {
  std::lock_guard<std::mutex> lock(getenv_mutex);
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str == "on" || str == "true" || str == "1";
}
