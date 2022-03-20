#include "triton/ir/Dialect.h"
#include "triton/ir/Types.h"

using namespace mlir;
using namespace mlir::triton;

// F8 & BF8
Float8Type Float8Type::get(MLIRContext *context) {
  return Base::get(context);
}

BFloat8Type BFloat8Type::get(MLIRContext *context) {
  return Base::get(context);
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//
struct triton::detail::PointerTypeStorage : public TypeStorage {
  using KeyTy = std::pair<Type, unsigned>;

  static PointerTypeStorage *construct(TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<PointerTypeStorage>()) PointerTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(pointeeType, addressSpace);
  }

  PointerTypeStorage(const KeyTy &key)
      : pointeeType(key.first), addressSpace(key.second) {}

  Type pointeeType;
  unsigned addressSpace;
};

PointerType PointerType::get(Type pointeeType) {
  return Base::get(pointeeType.getContext(), pointeeType, 0);
}

PointerType PointerType::get(Type pointeeType, unsigned addressSpace) {
    return Base::get(pointeeType.getContext(), pointeeType, addressSpace);
}

Type PointerType::getPointeeType() const  { return getImpl()->pointeeType; }

unsigned PointerType::getAddressSpace() const { return getImpl()->addressSpace; }

//===----------------------------------------------------------------------===//
// Triton Dialect
//===----------------------------------------------------------------------===//
void TritonDialect::registerTypes() {
  addTypes<Float8Type, BFloat8Type, PointerType>();
}
