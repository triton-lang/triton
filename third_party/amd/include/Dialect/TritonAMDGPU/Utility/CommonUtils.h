#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_COMMONUTILS_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_COMMONUTILS_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <numeric>

namespace mlir::triton::AMD {
SmallVector<scf::ForOp> getLeafForOps(triton::FuncOp funcOp);

class CoordinateMapper {
public:
  CoordinateMapper(llvm::ArrayRef<int64_t> layout) : layout(layout) {
    bounds.resize(layout.size());
    std::exclusive_scan(layout.rbegin(), layout.rend(), bounds.begin(), 1,
                        std::multiplies<>());
  }

  SmallVector<int64_t> map(int64_t index) {
    SmallVector<int64_t> coords(bounds.size(), 0);
    for (size_t i = 1; i < bounds.size(); ++i) {
      size_t d = bounds.size() - i;
      coords[d] = index / bounds[d];
      index = index % bounds[d];
    }
    coords[0] = index;
    std::reverse(coords.begin(), coords.end());
    return coords;
  }

  // TODO (Ravil): add C++
  template <typename T>
  static std::vector<std::vector<T>> cartesian(const std::vector<T> &ranges,
                                               const std::vector<T> &order) {
    assert(ranges.size() == order.size());
    auto imageSize =
        std::accumulate(ranges.begin(), ranges.end(), 1, std::multiplies{});
    auto product =
        std::vector<std::vector<T>>(imageSize, std::vector<T>(ranges.size()));

    auto strides = CoordinateMapper::getDeviders(ranges, order);
    for (size_t vec = 0; vec < imageSize; ++vec) {
      for (size_t elem = 0; elem < ranges.size(); ++elem) {
        product[vec][elem] = (vec / strides[elem]) % ranges[elem];
      }
    }

    return product;
  }

private:
  template <typename T>
  static std::vector<T> getDeviders(const std::vector<T> &dims,
                                    const std::vector<T> &order) {
    std::vector<T> orderedDims(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      orderedDims[i] = dims[order[i]];
    }
    std::vector<T> strides(dims.size());
    std::exclusive_scan(orderedDims.begin(), orderedDims.end(), strides.begin(),
                        static_cast<T>(1), std::multiplies<>());

    std::vector<T> orderedDeviders(dims.size());
    for (size_t d = 0; d < dims.size(); ++d) {
      orderedDeviders[d] = strides[order[d]];
    }
    return orderedDeviders;
  }

  llvm::ArrayRef<int64_t> layout;
  std::vector<int> bounds;
};

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_COMMONUTILS_H_
