#include "GF2LinearAlgebra.h"

#include "third_party/f2reduce/f2reduce.h"

#include <algorithm>
#include <cassert>

namespace mlir::triton::detail {

GF2Matrix::GF2Matrix(unsigned rows, unsigned columns)
    : rows(rows), columns(columns), stride(std::max(1u, (columns + 63) / 64)),
      storage(rows * stride, 0) {}

bool GF2Matrix::get(unsigned row, unsigned column) const {
  assert(row < rows && column < columns);
  return storage[row * stride + column / 64] & (uint64_t{1} << column % 64);
}

void GF2Matrix::set(unsigned row, unsigned column, bool value) {
  assert(row < rows && column < columns);
  uint64_t &word = storage[row * stride + column / 64];
  uint64_t mask = uint64_t{1} << column % 64;
  if (value)
    word |= mask;
  else
    word &= ~mask;
}

uint64_t GF2Matrix::getRow64(unsigned row) const {
  assert(row < rows && columns <= 64);
  return storage[row * stride];
}

void GF2Matrix::setRow64(unsigned row, uint64_t value) {
  assert(row < rows && columns <= 64);
  storage[row * stride] = value;
}

GF2Matrix GF2Matrix::rowReduced() const {
  GF2Matrix result = *this;
  if (rows && columns)
    f2reduce::inplace_rref_strided(result.storage.data(), rows, columns,
                                   stride);
  return result;
}

bool GF2Matrix::isZeroRow(unsigned row) const {
  assert(row < rows);
  for (unsigned word = 0; word < stride; ++word)
    if (storage[row * stride + word])
      return false;
  return true;
}

std::optional<unsigned> GF2Matrix::firstSetColumn(unsigned row) const {
  assert(row < rows);
  for (unsigned column = 0; column < columns; ++column)
    if (get(row, column))
      return column;
  return std::nullopt;
}

unsigned GF2Matrix::rank() const {
  GF2Matrix reduced = rowReduced();
  unsigned result = 0;
  for (unsigned row = 0; row < rows; ++row)
    result += !reduced.isZeroRow(row);
  return result;
}

llvm::SmallVector<unsigned> GF2Matrix::pivotColumns() const {
  GF2Matrix reduced = rowReduced();
  llvm::SmallVector<unsigned> result;
  for (unsigned row = 0; row < rows; ++row)
    if (auto pivot = reduced.firstSetColumn(row))
      result.push_back(*pivot);
  return result;
}

GF2Matrix GF2Matrix::nullspace() const {
  GF2Matrix reduced = rowReduced();
  llvm::SmallVector<bool> isPivot(columns, false);
  for (unsigned row = 0; row < rows; ++row)
    if (auto pivot = reduced.firstSetColumn(row))
      isPivot[*pivot] = true;

  GF2Matrix result(columns - std::count(isPivot.begin(), isPivot.end(), true),
                   columns);
  unsigned resultRow = 0;
  for (unsigned freeColumn = 0; freeColumn < columns; ++freeColumn) {
    if (isPivot[freeColumn])
      continue;
    result.set(resultRow, freeColumn);
    for (unsigned row = 0; row < rows; ++row)
      if (auto pivot = reduced.firstSetColumn(row);
          pivot && reduced.get(row, freeColumn))
        result.set(resultRow, *pivot);
    ++resultRow;
  }
  return result;
}

GF2Matrix GF2Matrix::coordinateComplement() const {
  llvm::SmallVector<bool> isPivot(columns, false);
  for (unsigned pivot : pivotColumns())
    isPivot[pivot] = true;

  GF2Matrix result(columns - std::count(isPivot.begin(), isPivot.end(), true),
                   columns);
  unsigned resultRow = 0;
  for (unsigned column = 0; column < columns; ++column)
    if (!isPivot[column])
      result.set(resultRow++, column);
  return result;
}

std::optional<GF2Matrix> GF2Matrix::solve(const GF2Matrix &rhs) const {
  assert(rows == rhs.rows && "matrix row counts must match");
  GF2Matrix augmented(rows, columns + rhs.columns);
  for (unsigned row = 0; row < rows; ++row) {
    for (unsigned column = 0; column < columns; ++column)
      augmented.set(row, column, get(row, column));
    for (unsigned column = 0; column < rhs.columns; ++column)
      augmented.set(row, columns + column, rhs.get(row, column));
  }
  augmented = augmented.rowReduced();

  GF2Matrix solution(columns, rhs.columns);
  for (unsigned row = 0; row < rows; ++row) {
    auto pivot = augmented.firstSetColumn(row);
    if (!pivot)
      continue;
    if (*pivot >= columns)
      return std::nullopt;
    for (unsigned rhsColumn = 0; rhsColumn < rhs.columns; ++rhsColumn)
      solution.set(*pivot, rhsColumn, augmented.get(row, columns + rhsColumn));
  }
  return solution;
}

} // namespace mlir::triton::detail
