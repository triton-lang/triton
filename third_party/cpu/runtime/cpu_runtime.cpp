#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <float.h>

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

using FLOAT16 = struct _FLOAT16 {
#ifdef FLT16_MAX
  _Float16 x;
#else
  uint16_t x;
#endif

  float toFloat32() const {
#ifdef FLT16_MAX
    return static_cast<float>(x);
#else
    // Based on https://gist.github.com/zhuker/b4bd1fb306c7b04975b712c37c4c4075
    uint32_t t1;
    uint32_t t2;
    uint32_t t3;

    t1 = x & 0x7fffu; // Non-sign bits
    t2 = x & 0x8000u; // Sign bit
    t3 = x & 0x7c00u; // Exponent

    t1 <<= 13u; // Align mantissa on MSB
    t2 <<= 16u; // Shift sign bit into position

    t1 += 0x38000000; // Adjust bias

    t1 = (t3 == 0 ? 0 : t1); // Denormals-as-zero

    t1 |= t2; // Re-insert sign bit

    float out;
    *((uint32_t *)&out) = t1;
    return out;
#endif
  }
};

struct FormatInfo {
  bool isInt;
  bool isSigned;
  int bitWidth;
  int maxIntDigits;
  bool hasNegative;
  bool scientific;
  bool isHex;
};

template <typename T> struct RawMemRefDescriptor {
  const T *allocated;
  const T *aligned;
  intptr_t offset;
  intptr_t sizesAndStrides[];
};

template <typename T> class MemRefDescriptor {
private:
  const T *data_;
  std::vector<intptr_t> sizes_;
  std::vector<intptr_t> strides_;

  MemRefDescriptor(const T *data, std::vector<intptr_t> sizes,
                   std::vector<intptr_t> strides)
      : data_(data), sizes_(std::move(sizes)), strides_(std::move(strides)) {}

public:
  MemRefDescriptor(int32_t rank, void *rawDescriptor) {
    auto *rawDesc = static_cast<RawMemRefDescriptor<T> *>(rawDescriptor);
    data_ = rawDesc->aligned + rawDesc->offset;
    sizes_.insert(sizes_.begin(), rawDesc->sizesAndStrides,
                  rawDesc->sizesAndStrides + rank);
    strides_.insert(strides_.begin(), rawDesc->sizesAndStrides + rank,
                    rawDesc->sizesAndStrides + rank * 2);
  }

  const T *data() const { return data_; }

  int64_t rank() const { return static_cast<int64_t>(sizes_.size()); }

  int64_t size(int64_t dim) const { return sizes_[dim]; }

  int64_t stride(int64_t dim) const { return strides_[dim]; }

  MemRefDescriptor<T> subView(int64_t idx) const {
    assert(rank() > 1);
    return {data_ + idx * stride(0),
            {sizes_.begin() + 1, sizes_.end()},
            {strides_.begin() + 1, strides_.end()}};
  }
};

struct UnrankedMemRefType {
  int64_t rank;
  void *descriptor;
};

template <typename T>
std::pair<int /* numDigits */, bool /* isNegative */> computeDigitInfo(T val) {
  if (val == 0)
    return {1, false};
  int digits =
      std::max(static_cast<int>(std::log10(val >= 0 ? val : -val)), 0) + 1;
  return {digits, val < 0};
}

template <>
std::pair<int /* numDigits */, bool /* isNegative */>
computeDigitInfo<FLOAT16>(FLOAT16 val) {
  return computeDigitInfo<float>(val.toFloat32());
}

template <typename T>
std::tuple<int, int, bool> computeDigitStats(const MemRefDescriptor<T> &desc) {
  int maxIntDigits = 0;
  int minIntDigits = std::numeric_limits<int>::max();
  bool hasNegative = false;

  if (desc.rank() == 1) {
    const T *data = desc.data();
    int64_t stride = desc.stride(0);
    for (int64_t i = 0; i < desc.size(0); ++i) {
      auto [digits, negative] = computeDigitInfo<T>(data[i * stride]);
      hasNegative |= negative;
      maxIntDigits = std::max(maxIntDigits, digits);
      minIntDigits = std::min(minIntDigits, digits);
    }
  } else {
    for (int64_t i = 0; i < desc.size(0); ++i) {
      auto [maxDigits, minDigits, negative] =
          computeDigitStats(desc.subView(i));
      hasNegative |= negative;
      maxIntDigits = std::max(maxIntDigits, maxDigits);
      minIntDigits = std::min(minIntDigits, minDigits);
    }
  }

  return std::make_tuple(maxIntDigits, minIntDigits, hasNegative);
}

template <typename T>
FormatInfo getFormatInfo(const MemRefDescriptor<T> &desc, bool isInt,
                         bool isSigned, int32_t bitWidth, bool isHex) {
  if (isHex) {
    assert(bitWidth >= 8 && bitWidth <= 64 && bitWidth % 8 == 0);
    return {isInt, isSigned, bitWidth, bitWidth / 4, false, false, true};
  }
  auto [maxIntDigits, minIntDigits, hasNegative] = computeDigitStats(desc);
  // Fallback to the scientific format for certain cases.
  bool scientific;
  if (isInt) {
    scientific = false;
  } else {
    scientific = maxIntDigits + 2 + (hasNegative ? 1 : 0) > MAX_FLOAT_WIDTH;
    scientific |= maxIntDigits - minIntDigits > 3;
  }
  return {isInt,       isSigned,   bitWidth, maxIntDigits,
          hasNegative, scientific, false};
}

template <typename T>
void printFormattedElement(std::stringstream &ss, T val,
                           const FormatInfo &formatInfo) {
  // Right now, the GPU's hex float doesn't work correctly. C++ has std::
  // hexfloat, but let's consider only hex integers for now.
  if (formatInfo.isHex && formatInfo.isInt) {
    ss << "0x" << std::hex << std::setw(formatInfo.maxIntDigits)
       << std::setfill('0') << val;
    return;
  }

  int padding = 0;
  auto [digits, negative] = computeDigitInfo(val);
  if (!negative && formatInfo.hasNegative)
    padding++;
  if (formatInfo.scientific) {
    ss << std::scientific << std::setw(MAX_FLOAT_WIDTH)
       << std::setprecision(FLOAT_PREC) << std::string(padding, ' ') << val;
  } else {
    padding += formatInfo.maxIntDigits - digits;
    ss << std::fixed << std::setprecision(FLOAT_PREC)
       << std::string(padding, ' ') << val;
  }
}

// int8_t is printed as char, so use int16_t instead.
template <>
void printFormattedElement<int8_t>(std::stringstream &ss, int8_t val,
                                   const FormatInfo &formatInfo) {
  printFormattedElement<int16_t>(ss, val, formatInfo);
}

template <>
void printFormattedElement<uint8_t>(std::stringstream &ss, uint8_t val,
                                    const FormatInfo &formatInfo) {
  printFormattedElement<uint16_t>(ss, val, formatInfo);
}

template <>
void printFormattedElement<FLOAT16>(std::stringstream &ss, FLOAT16 val,
                                    const FormatInfo &formatInfo) {
  printFormattedElement<float>(ss, val.toFloat32(), formatInfo);
}

template <typename T>
void printToStreamRecursive(const MemRefDescriptor<T> &desc,
                            std::stringstream &ss, const FormatInfo &formatInfo,
                            const std::string &linePrefix) {
  if (desc.rank() > 1) {
    ss << "[";
    for (int64_t i = 0; i < desc.size(0); ++i) {
      printToStreamRecursive(desc.subView(i), ss, formatInfo, linePrefix + " ");
      if (i != desc.size(0) - 1)
        ss << ",\n" << linePrefix << " ";
    }
    ss << "]";
    return;
  }

  const T *data = desc.data();
  int64_t stride = desc.stride(0);
  int64_t numElems = desc.size(0);

  ss << "[";
  if (numElems <= ELEMS_PER_LINE) {
    for (int i = 0; i < numElems; i++) {
      printFormattedElement(ss, data[i * stride], formatInfo);
      if (i != numElems - 1)
        ss << ", ";
    }
  } else {
    // TODO: Too many lines? Omit the middle lines.
    for (int i = 0; i < numElems; i++) {
      printFormattedElement(ss, data[i * stride], formatInfo);
      if (i == numElems - 1)
        break;
      if (i % ELEMS_PER_LINE == ELEMS_PER_LINE - 1) {
        ss << ",\n" << linePrefix << " ";
      } else {
        ss << ", ";
      }
    }
  }
  ss << "]";
}

template <typename T>
void printToStream(const MemRefDescriptor<T> &desc, std::stringstream &ss,
                   const FormatInfo &partialFormatInfo,
                   const std::string &linePrefix) {
  FormatInfo formatInfo = getFormatInfo<T>(
      desc, partialFormatInfo.isInt, partialFormatInfo.isSigned,
      partialFormatInfo.bitWidth, partialFormatInfo.isHex);
  printToStreamRecursive(desc, ss, formatInfo, linePrefix);
}

void printMemRef(std::stringstream &ss, int32_t rank, void *descriptor,
                 int32_t btw, bool isInteger, bool isSignedInteger, bool asHex,
                 const std::string &linePrefix) {

  FormatInfo partialFormat{.isInt = isInteger,
                           .isSigned = isSignedInteger,
                           .bitWidth = btw,
                           .isHex = asHex};
  if (!isInteger) {
    switch (btw) {
    case 64:
      printToStream(MemRefDescriptor<double>(rank, descriptor), ss,
                    partialFormat, linePrefix);
      return;
    case 32:
      printToStream(MemRefDescriptor<float>(rank, descriptor), ss,
                    partialFormat, linePrefix);
      return;
    case 16:
      printToStream(MemRefDescriptor<FLOAT16>(rank, descriptor), ss,
                    partialFormat, linePrefix);
      return;
    default:
      llvm_unreachable("Unsupported bitWidth");
    }
  }
  if (isSignedInteger) {
    switch (btw) {
    case 64:
      printToStream(MemRefDescriptor<int64_t>(rank, descriptor), ss,
                    partialFormat, linePrefix);
      return;
    case 32:
      printToStream(MemRefDescriptor<int32_t>(rank, descriptor), ss,
                    partialFormat, linePrefix);
      return;
    case 16:
      printToStream(MemRefDescriptor<int16_t>(rank, descriptor), ss,
                    partialFormat, linePrefix);
      return;
    case 8:
      printToStream(MemRefDescriptor<int8_t>(rank, descriptor), ss,
                    partialFormat, linePrefix);
      return;
    case 1:
      printToStream(MemRefDescriptor<bool>(rank, descriptor), ss, partialFormat,
                    linePrefix);
      return;
    default:
      llvm_unreachable("Unsupported bitWidth");
    }
  }
  switch (btw) {
  case 64:
    printToStream(MemRefDescriptor<uint64_t>(rank, descriptor), ss,
                  partialFormat, linePrefix);
    return;
  case 32:
    printToStream(MemRefDescriptor<uint32_t>(rank, descriptor), ss,
                  partialFormat, linePrefix);
    return;
  case 16:
    printToStream(MemRefDescriptor<uint16_t>(rank, descriptor), ss,
                  partialFormat, linePrefix);
    return;
  case 8:
    printToStream(MemRefDescriptor<uint8_t>(rank, descriptor), ss,
                  partialFormat, linePrefix);
    return;
  case 1:
    printToStream(MemRefDescriptor<bool>(rank, descriptor), ss, partialFormat,
                  linePrefix);
    return;
  default:
    llvm_unreachable("Unsupported bitWidth");
  }
}

} // namespace

extern "C" {

EXPORT void triton_assert(int32_t pid0, int32_t pid1, int32_t pid2, bool cond,
                          const char *message, const char *file, int32_t line,
                          const char *function) {
  if (cond)
    return;
  fprintf(stderr, "%s:%u: %s: block: [%u, %u, %u] Assertion `%s` failed.\n",
          file, line, function, pid0, pid1, pid2, message);
  abort();
}

// Print the pid prefix like the GPU and interpreter. And vectors are printed
// similar to Torch's printing like the following:
// (1, 0, 0) x: [ -0.4963,  -1.7682,   2.0885,   3.1320,  -4.3074,   5.6341,
//                -6.4901,   7.8964,  -8.4556,  -9.6323, -10.3489, -11.4017,
//               -12.0223,  13.1689,  14.2939, -15.5185]
EXPORT void triton_print_unranked_memref(int32_t pid0, int32_t pid1,
                                         int32_t pid2, const char *prefix,
                                         UnrankedMemRefType memref, int32_t btw,
                                         bool isInteger, bool isSigned,
                                         bool asHex) {
  std::stringstream ss;
  ss << "(" << pid0 << ", " << pid1 << ", " << pid2 << ")" << prefix;
  std::string linePrefix(ss.str().size(), ' ');
  printMemRef(ss, memref.rank, memref.descriptor, btw, isInteger, isSigned,
              asHex, linePrefix);
  ss << "\n";
  std::cout << ss.str() << std::flush;
}

} // extern "C"
