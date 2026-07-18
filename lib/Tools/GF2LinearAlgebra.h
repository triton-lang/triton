#ifndef TRITON_LIB_TOOLS_GF2LINEARALGEBRA_H
#define TRITON_LIB_TOOLS_GF2LINEARALGEBRA_H

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::triton::detail {

// A row-major matrix over GF(2). This is intentionally private to TritonTools:
// LinearLayout is the public representation, while this class centralizes the
// elimination used to reason about its bases.
class GF2Matrix {
public:
  GF2Matrix(unsigned rows, unsigned columns);

  unsigned getNumRows() const { return rows; }
  unsigned getNumColumns() const { return columns; }

  bool get(unsigned row, unsigned column) const;
  void set(unsigned row, unsigned column, bool value = true);

  // These helpers are for clients whose vector spaces fit in one word.
  uint64_t getRow64(unsigned row) const;
  void setRow64(unsigned row, uint64_t value);

  GF2Matrix rowReduced() const;
  unsigned rank() const;
  llvm::SmallVector<unsigned> pivotColumns() const;

  // Return matrices whose rows are bases for the nullspace and for the
  // coordinate-vector complement of the row space, respectively.
  GF2Matrix nullspace() const;
  GF2Matrix coordinateComplement() const;

  // Solve this * X = rhs, choosing zero for every free variable.
  std::optional<GF2Matrix> solve(const GF2Matrix &rhs) const;

private:
  bool isZeroRow(unsigned row) const;
  std::optional<unsigned> firstSetColumn(unsigned row) const;

  unsigned rows;
  unsigned columns;
  unsigned stride;
  llvm::SmallVector<uint64_t> storage;
};

} // namespace mlir::triton::detail

#endif // TRITON_LIB_TOOLS_GF2LINEARALGEBRA_H
