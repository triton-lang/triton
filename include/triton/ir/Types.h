#ifndef TRITON_IR_TYPES_H_
#define TRITON_IR_TYPES_H_

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace triton {

namespace detail {
struct PointerTypeStorage;
} // namespace detail

// TODO: Should be base class be FloatType?
class Float8Type : public Type::TypeBase<Float8Type, Type, TypeStorage> {
public:
  using Base::Base;

  static Float8Type get(MLIRContext *context);
};

class BFloat8Type : public Type::TypeBase<BFloat8Type, Type, TypeStorage> {
public:
  using Base::Base;

  static BFloat8Type get(MLIRContext *context);
};

class PointerType : public Type::TypeBase<PointerType, Type, 
                                          detail::PointerTypeStorage> {
public:
  using Base::Base;

  static PointerType get(Type pointeeType);

  static PointerType get(Type pointeeType, unsigned addressSpace);

  Type getPointeeType() const;

  unsigned getAddressSpace() const;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_IR_TYPES_H_
