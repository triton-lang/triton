#include "triton/lang/encoding.h"

#include <climits>
#include <codecvt>
#include <locale>
#include <iostream>


static void Append16LE(std::string& str, char16_t c) {
  str.push_back(c & UCHAR_MAX);
  str.push_back((c >> 8) & UCHAR_MAX);
}


static void Append32LE(std::string& str, char32_t c) {
  Append16LE(str, c & USHRT_MAX);
  Append16LE(str, (c >> 16) & USHRT_MAX);
}


void ConvertToUTF16(std::string& str) {
  std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t> utf8_ucs2_cvt;
  auto str16 = utf8_ucs2_cvt.from_bytes(str);
  str.resize(0);
  for (auto c16: str16)
    Append16LE(str, c16);
}


void ConvertToUTF32(std::string& str) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> utf8_ucs4_cvt;
  auto str32 = utf8_ucs4_cvt.from_bytes(str);
  str.resize(0);
  for (auto c32: str32)
    Append32LE(str, c32);
}


void AppendUCN(std::string& str, int c) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> utf8_ucs4_cvt;
  str += utf8_ucs4_cvt.to_bytes(static_cast<char32_t>(c));
}
