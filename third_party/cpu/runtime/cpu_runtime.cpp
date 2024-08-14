#include <cassert>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT
#endif

namespace {

// A poor man's Torch-like pretty print for tensors and vectors.
const int MAX_FLOAT_WIDTH = 8;
const int FLOAT_PREC = 4;
const int ELEMS_PER_LINE = 8;

struct FormatInfo {
  bool isInt;
  int bitWidth;
  int maxIntDigits;
  bool hasNegative;
  bool scientific;
};

template <typename T>
std::pair<int /* numDigits */, bool /* isNegative */>
computeDigitInfoHelper(const void *array, size_t index) {
  T elem = static_cast<const T *>(array)[index];
  if (elem == 0)
    return {1, false};
  return {static_cast<int>(std::log10(std::abs(elem))) + 1, elem < 0};
}

std::pair<int, bool> computeDigitInfo(void *vec, int32_t isInt,
                                      int32_t bitWidth, size_t index) {

  if (isInt == 0) {
    if (bitWidth == 32)
      return computeDigitInfoHelper<float>(vec, index);
    else if (bitWidth == 64)
      return computeDigitInfoHelper<double>(vec, index);
    else
      assert(false && "Unsupported bitWidth");
  } else {
    // TODO: Handle signed types?
    if (bitWidth == 64)
      return computeDigitInfoHelper<int64_t>(vec, index);
    else if (bitWidth == 32)
      return computeDigitInfoHelper<int32_t>(vec, index);
    else if (bitWidth == 16)
      return computeDigitInfoHelper<int16_t>(vec, index);
    else if (bitWidth == 8)
      return computeDigitInfoHelper<int8_t>(vec, index);
    else
      assert(false && "Unsupported bitWidth");
  }
}

FormatInfo getFormatInfo(void *vec, bool isInt, int32_t bitWidth,
                         int64_t numElem) {
  // Compute the max/min widths for pretty printing.
  int maxIntDigits = 0;
  int minIntDigits = std::numeric_limits<int>::max();
  bool hasNegative = false;
  for (int64_t i = 0; i < numElem; ++i) {
    auto [digits, negative] = computeDigitInfo(vec, isInt, bitWidth, i);
    hasNegative |= negative;
    maxIntDigits = std::max(maxIntDigits, digits);
    minIntDigits = std::min(minIntDigits, digits);
  }
  // Fallback to the scientific format for certain cases.
  bool scientific;
  if (isInt) {
    scientific = false;
  } else {
    scientific = maxIntDigits + 2 + (hasNegative ? 1 : 0) > MAX_FLOAT_WIDTH;
    scientific |= maxIntDigits - minIntDigits > 3;
  }
  return {isInt, bitWidth, maxIntDigits, hasNegative, scientific};
}

template <typename T>
void printElementHelper(std::stringstream &ss, const void *array,
                        size_t index) {
  ss << static_cast<const T *>(array)[index];
}

void printElement(std::stringstream &ss, const void *vec, size_t index,
                  bool isInt, int bitWidth) {
  if (isInt == 0) {
    switch (bitWidth) {
    case 32:
      printElementHelper<float>(ss, vec, index);
      break;
    case 64:
      printElementHelper<double>(ss, vec, index);
      break;
    default:
      assert(false && "Unsupported bitWidth");
    }
  } else {
    switch (bitWidth) {
    case 64:
      printElementHelper<int64_t>(ss, vec, index);
      break;
    case 32:
      printElementHelper<int32_t>(ss, vec, index);
      break;
    case 16:
      printElementHelper<int16_t>(ss, vec, index);
      break;
    case 8:
      // TODO: Seems like not working well. Need to fix it.
      printElementHelper<int8_t>(ss, vec, index);
      break;
    default:
      assert(false && "Unsupported bitWidth");
    }
  }
}

void printFormattedElement(std::stringstream &ss, void *vec, size_t index,
                           const FormatInfo &formatInfo) {
  int padding = 0;
  auto [digits, negative] =
      computeDigitInfo(vec, formatInfo.isInt, formatInfo.bitWidth, index);
  if (!negative && formatInfo.hasNegative)
    padding++;
  if (formatInfo.scientific) {
    ss << std::scientific << std::setw(MAX_FLOAT_WIDTH)
       << std::setprecision(FLOAT_PREC) << std::string(padding, ' ');
    printElement(ss, vec, index, formatInfo.isInt, formatInfo.bitWidth);
  } else {
    padding += formatInfo.maxIntDigits - digits;
    ss << std::fixed << std::setprecision(FLOAT_PREC)
       << std::string(padding, ' ');
    printElement(ss, vec, index, formatInfo.isInt, formatInfo.bitWidth);
  }
}
} // namespace

extern "C" {

EXPORT void triton_assert(bool cond, char *c) {
  if (!cond)
    fprintf(stderr, "%s\n", c);
}

// Print the pid prefix like the GPU ad interpreter. And vectors are printed
// similar to Torch's printing like the following:
// (1, 0, 0) x: [ -0.4963,  -1.7682,   2.0885,   3.1320,  -4.3074,   5.6341,
//                -6.4901,   7.8964,  -8.4556,  -9.6323, -10.3489, -11.4017,
//               -12.0223,  13.1689,  14.2939, -15.5185]
//
// TODO: Implement for higher dimension vectors.
EXPORT void triton_vector_print(int32_t pid0, int32_t pid1, int32_t pid2,
                                const char *prefix, void *vec, int32_t isInt,
                                int32_t bitWidth, int64_t numElem) {

  FormatInfo formatInfo = getFormatInfo(vec, isInt != 0, bitWidth, numElem);

  std::stringstream ss;
  ss << "(" << pid0 << ", " << pid1 << ", " << pid2 << ")" << prefix << "[";
  const size_t header = ss.str().size();

  if (numElem <= ELEMS_PER_LINE) {
    for (int i = 0; i < numElem; i++) {
      printFormattedElement(ss, vec, i, formatInfo);
      if (i != numElem - 1)
        ss << ", ";
    }
  } else {
    // TODO: Too many lines? Omit the middle lines.
    for (int i = 0; i < numElem; i++) {
      printFormattedElement(ss, vec, i, formatInfo);
      if (i == numElem - 1)
        break;
      if (i % ELEMS_PER_LINE == ELEMS_PER_LINE - 1) {
        ss << ",\n" << std::string(header, ' ');
      } else {
        ss << ", ";
      }
    }
  }
  ss << "]\n";
  std::cout << ss.str() << std::flush;
}

} // extern "C"
