#pragma once

#ifndef _WGTCC_TYPE_H_
#define _WGTCC_TYPE_H_

#include "mem_pool.h"
#include "scope.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <list>


class Scope;
class Token;
class Expr;

class Type;
class QualType;
class VoidType;
class Identifier;
class Object;
class Constant;

class ArithmType;
class DerivedType;
class ArrayType;
class TileType;
class FuncType;
class PointerType;
class StructType;
class EnumType;


enum {
  // Storage class specifiers
  S_TYPEDEF = 0x01,
  S_EXTERN = 0x02,
  S_STATIC = 0x04,
  S_THREAD = 0x08,
  S_CONSTANT = 0x10,
  S_GLOBAL = 0x20,

  // Type specifier
  T_SIGNED = 0x40,
  T_UNSIGNED = 0x80,
  T_CHAR = 0x100,
  T_SHORT = 0x200,
  T_INT = 0x400,
  T_LONG = 0x800,
  T_VOID = 0x1000,
  T_HALF = 0x2000,
  T_FLOAT = 0x4000,
  T_DOUBLE = 0x8000,
  T_BOOL = 0x10000,
  T_COMPLEX = 0x20000,
  // T_ATOMIC = 0x40000,
  T_STRUCT_UNION = 0x80000,
  T_ENUM = 0x100000,
  T_TYPEDEF_NAME = 0x200000,

  T_LLONG = 0x4000000,

  // Function specifier
  F_INLINE = 0x8000000,
  F_NORETURN = 0x10000000,
};


struct Qualifier {
  enum {
    CONST = 0x01,
    RESTRICT = 0x02,
    VOLATILE = 0x04,
    CMEM = 0x08,
    MASK = CONST | RESTRICT | VOLATILE | CMEM
  };
};


class QualType {
public:
  QualType(Type* ptr, int quals=0x00)
      : ptr_(reinterpret_cast<intptr_t>(ptr)) {
    assert((quals & ~Qualifier::MASK) == 0);
    ptr_ |= quals;
  }

  operator bool() const { return !IsNull(); }
  bool IsNull() const { return GetPtr() == nullptr; }
  const Type* GetPtr() const {
    return reinterpret_cast<const Type*>(ptr_ & ~Qualifier::MASK);
  }
  Type* GetPtr() {
    return reinterpret_cast<Type*>(ptr_ & ~Qualifier::MASK);
  }
  Type& operator*() { return *GetPtr(); }
  const Type& operator*() const { return *GetPtr(); }
  Type* operator->() { return GetPtr(); }
  const Type* operator->() const { return GetPtr(); }

  // Indicate whether the specified types are identical(exclude qualifiers).
  friend bool operator==(QualType lhs, QualType rhs) {
    return lhs.operator->() == rhs.operator->();
  }
  friend bool operator!=(QualType lhs, QualType rhs) {
    return !(lhs == rhs);
  }

  int Qual() const { return ptr_ & 0x07; }
  bool IsConstQualified() const { return ptr_ & Qualifier::CONST; }
  bool IsRestrictQualified() const { return ptr_ & Qualifier::RESTRICT; }
  bool IsVolatileQualified() const { return ptr_ & Qualifier::VOLATILE; }
  bool IsConstantQualified() const { return ptr_ & Qualifier::CMEM; }

private:
  intptr_t ptr_;
};


class Type {
public:
  static const int intWidth_ = 4;
  static const int machineWidth_ = 8;

  bool operator!=(const Type& other) const = delete;
  bool operator==(const Type& other) const = delete;

  virtual bool Compatible(const Type& other) const {
    return complete_ == other.complete_;
  }

  virtual ~Type() {}

  // For Debugging
  virtual std::string Str() const = 0;
  virtual int Width() const = 0;
  virtual int Align() const { return Width(); }
  static int MakeAlign(int offset, int align) {
    if ((offset % align) == 0)
      return offset;
    if (offset >= 0)
      return offset + align - (offset % align);
    else
      return offset - align - (offset % align);
  }

  static QualType MayCast(QualType type, bool inProtoScope=false);
  bool Complete() const { return complete_; }
  void SetComplete(bool complete) const { complete_ = complete; }

  bool IsReal() const { return IsInteger() || IsFloat(); };
  virtual bool IsScalar() const { return false; }
  virtual bool IsFloat() const { return false; }
  virtual bool IsInteger() const { return false; }
  virtual bool IsBool() const { return false; }
  virtual bool IsVoidPointer() const { return false; }
  virtual bool IsUnsigned() const { return false; }
  virtual bool IsTile() const { return ToTile() != nullptr; }

  const Type* ScalarType() const;
  Type* ScalarType();

  virtual VoidType*           ToVoid() { return nullptr; }
  virtual const VoidType*     ToVoid() const { return nullptr; }
  virtual ArithmType*         ToArithm() { return nullptr; }
  virtual const ArithmType*   ToArithm() const { return nullptr; }
  virtual ArrayType*          ToArray() { return nullptr; }
  virtual const ArrayType*    ToArray() const { return nullptr; }
  virtual TileType*           ToTile() { return nullptr; }
  virtual const TileType*     ToTile() const { return nullptr; }
  virtual FuncType*           ToFunc() { return nullptr; }
  virtual const FuncType*     ToFunc() const { return nullptr; }
  virtual PointerType*        ToPointer() { return nullptr; }
  virtual const PointerType*  ToPointer() const { return nullptr; }
  virtual DerivedType*        ToDerived() { return nullptr; }
  virtual const DerivedType*  ToDerived() const { return nullptr; }
  virtual StructType*         ToStruct() { return nullptr; }
  virtual const StructType*   ToStruct() const { return nullptr; }

protected:
  Type(MemPool* pool, bool complete)
      : complete_(complete), pool_(pool) {}

  mutable bool complete_;
  MemPool* pool_;
};


class VoidType : public Type {
public:
  static VoidType* New();
  virtual ~VoidType() {}
  virtual VoidType* ToVoid() { return this; }
  virtual const VoidType* ToVoid() const { return this; }
  virtual bool Compatible(const Type& other) const { return other.ToVoid(); }
  virtual int Width() const {
    // Non-standard GNU extension
    return 1;
  }
  virtual std::string Str() const { return "void:1"; }

protected:
  explicit VoidType(MemPool* pool): Type(pool, false) {}
};


class ArithmType : public Type {
public:
  static ArithmType* New(int typeSpec);

  virtual ~ArithmType() {}
  virtual ArithmType* ToArithm() { return this; }
  virtual const ArithmType* ToArithm() const { return this; }
  virtual bool Compatible(const Type& other) const {
    // C11 6.2.7 [1]: Two types have compatible type if their types are the same
    // But I would to loose this constraints: integer and pointer are compatible
    // if (IsInteger() && other.ToPointer())
    //   return other.Compatible(*this);
    return this == &other;
  }

  virtual int Width() const;
  virtual std::string Str() const;
  virtual bool IsScalar() const { return true; }
  virtual bool IsInteger() const { return !IsFloat() && !IsComplex(); }
  virtual bool IsUnsigned() const { return tag_ & T_UNSIGNED; }
  virtual bool IsFloat() const {
    return (tag_ & T_HALF) || (tag_ & T_FLOAT) || (tag_ & T_DOUBLE);
  }
  virtual bool IsBool() const { return tag_ & T_BOOL; }
  bool IsComplex() const { return tag_ & T_COMPLEX; }
  int Tag() const { return tag_; }
  int Rank() const;
  static ArithmType* IntegerPromote(ArithmType* type) {
    assert(type->IsInteger());
    if (type->Rank() < ArithmType::New(T_INT)->Rank())
      return ArithmType::New(T_INT);
    return type;
  }
  static ArithmType* MaxType(ArithmType* lhsType,
                                   ArithmType* rhsType);

protected:
  explicit ArithmType(MemPool* pool, int spec)
    : Type(pool, true), tag_(Spec2Tag(spec)) {}

private:
  static int Spec2Tag(int spec);

  int tag_;
};


class DerivedType : public Type {
public:
  QualType Derived() const { return derived_; }
  void SetDerived(QualType derived) { derived_ = derived; }
  virtual DerivedType* ToDerived() { return this; }
  virtual const DerivedType* ToDerived() const { return this; }

protected:
  DerivedType(MemPool* pool, QualType derived)
      : Type(pool, true), derived_(derived) {}

  QualType derived_;
};


class PointerType : public DerivedType {
public:
  static PointerType* New(QualType derived);
  virtual ~PointerType() {}
  virtual PointerType* ToPointer() { return this; }
  virtual const PointerType* ToPointer() const { return this; }
  virtual bool Compatible(const Type& other) const;
  virtual int Width() const { return 8; }
  virtual bool IsScalar() const { return true; }
  virtual bool IsVoidPointer() const { return derived_->ToVoid(); }
  virtual std::string Str() const {
    return derived_->Str() + "*:" + std::to_string(Width());
  }

protected:
  PointerType(MemPool* pool, QualType derived): DerivedType(pool, derived) {}
};


class ArrayType : public DerivedType {
public:
  static ArrayType* New(int len, QualType eleType);
  static ArrayType* New(Expr* expr, QualType eleType);
  virtual ~ArrayType() { /*delete derived_;*/ }

  virtual ArrayType* ToArray() { return this; }
  virtual const ArrayType* ToArray() const { return this; }
  virtual bool Compatible(const Type& other) const;
  virtual int Width() const {
    return Complete() ? (derived_->Width() * len_): 0;
  }
  virtual int Align() const { return derived_->Align(); }
  virtual std::string Str() const {
    return derived_->Str() + "[]:" + std::to_string(Width());
  }

  int GetElementOffset(int idx) const { return derived_->Width() * idx; }
  int Len() const { return len_; }
  void SetLen(int len) { len_ = len; }
  bool Variadic() const { return lenExpr_ != nullptr; }

protected:
  ArrayType(MemPool* pool, Expr* lenExpr, QualType derived)
      : DerivedType(pool, derived),
        lenExpr_(lenExpr), len_(0) {
    SetComplete(false);
  }

  ArrayType(MemPool* pool, int len, QualType derived)
      : DerivedType(pool, derived),
        lenExpr_(nullptr), len_(len) {
    SetComplete(len_ >= 0);
  }
  const Expr* lenExpr_;
  int len_;
};

class TileType : public DerivedType {
public:
  using ShapeExpr = std::vector<Expr*>;
  using ShapeInt = std::vector<int>;

public:
  static TileType* New(const ShapeExpr& expr, QualType eleType);
  static TileType* New(const ShapeInt& shape, QualType eleType);
  virtual ~TileType() { }

  virtual TileType* ToTile() { return this; }
  virtual const TileType* ToTile() const { return this; }
  virtual bool Compatible(const Type& other) const;
  virtual int Width() const { return Complete() ? derived_->Width()*NumEle() : 0; }
  virtual int Align() const { return derived_->Align(); }
  virtual std::string Str() const {
    return derived_->Str() + "[{}]:" + std::to_string(Width());
  }

  ShapeInt Shape() { return shape_; }
  int NumEle() const {
    int ret = 1;
    for(int s: shape_)
      ret *= s;
    return ret;
  }

protected:
  TileType(MemPool* pool, const ShapeExpr& expr, QualType derived)
    : DerivedType(pool, derived),
      shapeExpr_(expr) {
    bool isComplete = true;
    for(Expr* s: shapeExpr_)
      isComplete = isComplete && !s;
    SetComplete(isComplete);
  }

  TileType(MemPool* pool, const ShapeInt& shape, QualType derived)
    : DerivedType(pool, derived),
      shape_(shape) {
    bool isComplete = true;
    for(int s: shape_)
      isComplete = isComplete && (s>=0);
    SetComplete(isComplete);
  }

protected:
  ShapeExpr shapeExpr_;
  ShapeInt shape_;
};

class FuncType : public DerivedType {
public:
  using ParamList = std::vector<Object*>;

public:
  static FuncType* New(QualType derived,
                       int funcSpec,
                       bool variadic,
                       const ParamList& params);
  virtual ~FuncType() {}
  virtual FuncType* ToFunc() { return this; }
  virtual const FuncType* ToFunc() const { return this; }
  virtual bool Compatible(const Type& other) const;
  virtual int Width() const { return 1; }
  virtual std::string Str() const;
  const ParamList& Params() const { return params_; }
  void SetParams(const ParamList& params) { params_ = params; }
  bool Variadic() const { return variadic_; }
  bool IsInline() const { return inlineNoReturn_ & F_INLINE; }
  bool IsNoReturn() const { return inlineNoReturn_ & F_NORETURN; }

protected:
  FuncType(MemPool* pool, QualType derived, int inlineReturn,
           bool variadic, const ParamList& params)
      : DerivedType(pool, derived), inlineNoReturn_(inlineReturn),
        variadic_(variadic), params_(params) {
    SetComplete(false);
  }

private:
  int inlineNoReturn_;
  bool variadic_;
  ParamList params_;
};


class StructType : public Type {
public:
  using MemberList = std::list<Object*>;
  using Iterator = std::list<Object*>::iterator;

public:
  static StructType* New(bool isStruct,
                         bool hasTag,
                         Scope* parent);
  virtual ~StructType() {}
  virtual StructType* ToStruct() { return this; }
  virtual const StructType* ToStruct() const { return this; }
  virtual bool Compatible(const Type& other) const;
  virtual int Width() const { return width_; }
  virtual int Align() const { return align_; }
  virtual std::string Str() const;

  // struct/union
  void AddMember(Object* member);
  void AddBitField(Object* member, int offset);
  bool IsStruct() const { return isStruct_; }
  Object* GetMember(const std::string& member);
  Scope* MemberMap() { return memberMap_; }
  MemberList& Members() { return members_; }
  int Offset() const { return offset_; }
  bool HasTag() const { return hasTag_; }
  void MergeAnony(Object* anony);
  void Finalize();

protected:
  // Default is incomplete
  StructType(MemPool* pool, bool isStruct, bool hasTag, Scope* parent);

  StructType(const StructType& other);

private:
  void CalcWidth();

  bool isStruct_;
  bool hasTag_;
  Scope* memberMap_;

  MemberList members_;
  int offset_;
  int width_;
  int align_;
  int bitFieldAlign_;
};

#endif
