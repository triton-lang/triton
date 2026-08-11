#ifndef TRITON_ANALYSIS_MEMORYFRONTIER_H
#define TRITON_ANALYSIS_MEMORYFRONTIER_H

#include "triton/Analysis/BufferRegion.h"

#include <array>
#include <map>
#include <utility>

namespace mlir::triton {

/// Read and write frontiers keyed by exact physical buffer regions. Each access
/// is associated with the scopes in which it remains live.
template <typename ScopeMaskT> class ScopedMemoryFrontier {
public:
  using AccessT = BufferRegionAccess;
  using AccessMap = std::map<AccessT, ScopeMaskT>;

  void addRead(AccessT access, ScopeMaskT scopes) {
    add(/*isWrite=*/false, std::move(access), scopes);
  }
  void addWrite(AccessT access, ScopeMaskT scopes) {
    add(/*isWrite=*/true, std::move(access), scopes);
  }

  ScopedMemoryFrontier &join(const ScopedMemoryFrontier &other) {
    for (unsigned i = 0; i < accesses.size(); ++i)
      for (const auto &[access, scopes] : other.accesses[i])
        accesses[i][access] |= scopes;
    return *this;
  }

  void join(const ScopedMemoryFrontier &other, ScopeMaskT scopes) {
    for (unsigned i = 0; i < accesses.size(); ++i)
      for (const auto &[access, accessScopes] : other.accesses[i])
        if (ScopeMaskT activeScopes = accessScopes & scopes)
          accesses[i][access] |= activeScopes;
  }

  void eraseScopes(ScopeMaskT scopes) {
    for (AccessMap &map : accesses)
      for (auto it = map.begin(); it != map.end();)
        if (it->second &= ~scopes)
          ++it;
        else
          it = map.erase(it);
  }

  bool hasHazard(const ScopedMemoryFrontier &other, ScopeMaskT scope) const {
    return intersects(other, /*lhsWrite=*/true, /*rhsWrite=*/false, scope) ||
           intersects(other, /*lhsWrite=*/false, /*rhsWrite=*/true, scope) ||
           intersects(other, /*lhsWrite=*/true, /*rhsWrite=*/true, scope);
  }

  template <typename Transform> void transformAccesses(Transform transform) {
    for (AccessMap &map : accesses) {
      AccessMap transformed;
      for (const auto &[access, scopes] : map)
        transformed[transform(access)] |= scopes;
      map = std::move(transformed);
    }
  }

  bool operator==(const ScopedMemoryFrontier &other) const {
    return accesses == other.accesses;
  }

private:
  void add(bool isWrite, AccessT access, ScopeMaskT scopes) {
    accesses[isWrite][std::move(access)] |= scopes;
  }

  bool intersects(const ScopedMemoryFrontier &other, bool lhsWrite,
                  bool rhsWrite, ScopeMaskT scope) const {
    for (const auto &[left, leftScopes] : accesses[lhsWrite]) {
      if (!(leftScopes & scope))
        continue;
      for (const auto &[right, rightScopes] : other.accesses[rhsWrite])
        if ((rightScopes & scope) &&
            (!left || !right || left->intersects(*right)))
          return true;
    }
    return false;
  }

  std::array<AccessMap, 2> accesses;
};

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_MEMORYFRONTIER_H
