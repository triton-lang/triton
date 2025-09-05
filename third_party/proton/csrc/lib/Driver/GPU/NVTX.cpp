#include "Driver/GPU/NVTX.h"

#include <codecvt>
#include <cstdint>
#include <locale>

namespace proton {

namespace {
// Convert wide-character messages to UTF-8 std::string
static std::string wideToUtf8(const wchar_t *wstr) {
  if (!wstr)
    return {};
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  return conv.to_bytes(wstr);
}

// Declare nvtx function params without including the nvtx header
struct RangePushAParams {
  const char *message;
};

struct RangePushWParams {
  const wchar_t *message;
};

} // namespace

namespace nvtx {

std::string getMessageFromRangePushA(const void *params) {
  if (const auto *p = static_cast<const RangePushAParams *>(params))
    return std::string(p->message ? p->message : "");
  return "";
}

std::string getMessageFromRangePushW(const void *params) {
  if (const auto *p = static_cast<const RangePushWParams *>(params)) {
    return wideToUtf8(p->message ? p->message : L"");
  }
  return "";
}

} // namespace nvtx

} // namespace proton