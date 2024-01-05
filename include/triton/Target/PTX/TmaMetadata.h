/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_TARGET_PTX_TMAMETADATA_H
#define TRITON_TARGET_PTX_TMAMETADATA_H

#include "python/triton/third_party/cuda/include/cuda.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include <map>
#include <utility>
#include <vector>

namespace mlir {
namespace triton {
namespace gpu {

struct TMAInfo {
  // --------------------------------------------
  // information to be filled into CUtensorMaps
  int tensorDataType;

  uint32_t tensorRank;

  // the argument indices for the runtime to get globalAddresses
  size_t globalAddressArgIdx;

  // the argument indices for the runtime to get globalDims, -1 stands for this
  // dim is padded
  std::vector<int32_t> globalDimsArgIdx;

  // the argument indices for the runtime to get globalStrides, -1 stands for
  // this dim is padded the runtime need to map the value to internal format
  std::vector<int32_t> globalStridesArgIdx;

  std::vector<uint32_t> boxDims;

  std::vector<uint32_t> elementStrides;

  int interleave;

  int swizzle;

  int l2Promotion;

  int oobFill;

  // --------------------------------------------
  // the argument indices for the runtime to send the address of tma_desc to the
  // binary
  int TMADescArgIdx;

  template <typename T>
  void dump_vec(const std::vector<T> &vec, llvm::StringRef info) const {
    llvm::errs() << info << ": ";
    for (const T &e : vec)
      llvm::errs() << e << ",";
    llvm::errs() << "\n";
  }

  void dump() const {
    llvm::errs() << "TMA Info: ----------"
                 << "\n";
    llvm::errs() << "-- tensorDataType: " << tensorDataType
                 << ", tensorRank: " << tensorRank << "\n";
    llvm::errs() << "-- globalAddressArgIdx: " << globalAddressArgIdx << "\n";
    llvm::errs() << "-- TMADescArgIdx: " << TMADescArgIdx << "\n";
    dump_vec<int32_t>(globalDimsArgIdx, "-- globalDimsArgIdx");
    dump_vec<int32_t>(globalStridesArgIdx, "-- globalStridesArgIdx");
    dump_vec<uint32_t>(boxDims, "-- boxDims");
    dump_vec<uint32_t>(elementStrides, "-- elementStrides");
    llvm::errs() << "-- interleave: " << interleave << "\n";
    llvm::errs() << "-- swizzle: " << swizzle << "\n";
    llvm::errs() << "-- l2Promotion: " << l2Promotion << "\n";
    llvm::errs() << "-- oobFill: " << oobFill << "\n";
  };
};

using TMAMetadataTy = std::vector<TMAInfo>;

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_TARGET_PTX_TMAMETADATA_H
