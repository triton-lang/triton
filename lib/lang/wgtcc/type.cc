#include "triton/lang/wgtcc/type.h"

#include "triton/lang/wgtcc/ast.h"
#include "triton/lang/wgtcc/scope.h"
#include "triton/lang/wgtcc/token.h"

#include <cassert>
#include <algorithm>
#include <iostream>


static MemPoolImp<VoidType>     voidTypePool;
static MemPoolImp<ArrayType>    arrayTypePool;
static MemPoolImp<TileType>     tileTypePool;
static MemPoolImp<FuncType>     funcTypePool;
static MemPoolImp<PointerType>  pointerTypePool;
static MemPoolImp<StructType>   structUnionTypePool;
static MemPoolImp<ArithmType>   arithmTypePool;


QualType Type::MayCast(QualType type, bool inProtoScope) {
  auto funcType = type->ToFunc();
  auto arrayType = type->ToArray();
  if (funcType) {
    return PointerType::New(funcType);
  } else if (arrayType) {
    auto ret = PointerType::New(arrayType->Derived());
    // C11 6.7.6.3 [7]: qualifiers are specified in '[]'
    // As we do not support qualifiers in '[]', the qualifier whould be none
    return QualType(ret, inProtoScope? 0: Qualifier::CONST);
  }
  return type;
}


VoidType* VoidType::New() {
  static auto ret = new (voidTypePool.Alloc()) VoidType(&voidTypePool);
  return ret;
}


ArithmType* ArithmType::New(int typeSpec) {
#define NEW_TYPE(tag)                                           \
  new (arithmTypePool.Alloc()) ArithmType(&arithmTypePool, tag);

  static auto boolType    = NEW_TYPE(T_BOOL);
  static auto charType    = NEW_TYPE(T_CHAR);
  static auto ucharType   = NEW_TYPE(T_UNSIGNED | T_CHAR);
  static auto shortType   = NEW_TYPE(T_SHORT);
  static auto ushortType  = NEW_TYPE(T_UNSIGNED | T_SHORT);
  static auto intType     = NEW_TYPE(T_INT);
  static auto uintType    = NEW_TYPE(T_UNSIGNED | T_INT);
  static auto longType    = NEW_TYPE(T_LONG);
  static auto ulongType   = NEW_TYPE(T_UNSIGNED | T_LONG);
  static auto llongType   = NEW_TYPE(T_LLONG)
  static auto ullongType  = NEW_TYPE(T_UNSIGNED | T_LLONG);
  static auto halfType    = NEW_TYPE(T_HALF);
  static auto floatType   = NEW_TYPE(T_FLOAT);
  static auto doubleType  = NEW_TYPE(T_DOUBLE);
  static auto ldoubleType = NEW_TYPE(T_LONG | T_DOUBLE);

  auto tag = ArithmType::Spec2Tag(typeSpec);
  switch (tag) {
  case T_BOOL:              return boolType;
  case T_CHAR:              return charType;
  case T_UNSIGNED | T_CHAR: return ucharType;
  case T_SHORT:             return shortType;
  case T_UNSIGNED | T_SHORT:return ushortType;
  case T_INT:               return intType;
  case T_UNSIGNED:
  case T_UNSIGNED | T_INT:  return uintType;
  case T_LONG:              return longType;
  case T_UNSIGNED | T_LONG: return ulongType;
  case T_LLONG:             return llongType;
  case T_UNSIGNED | T_LLONG:return ullongType;
  case T_HALF:              return halfType;
  case T_FLOAT:             return floatType;
  case T_DOUBLE:            return doubleType;
  case T_LONG | T_DOUBLE:   return ldoubleType;
  default:
    assert(tag & T_COMPLEX);
    Error("complex not supported yet");
  }
  return nullptr; // Make compiler happy

#undef NEW_TYPE
}


ArrayType* ArrayType::New(int len, QualType eleType) {
  return new (arrayTypePool.Alloc())
         ArrayType(&arrayTypePool, len, eleType);
}


ArrayType* ArrayType::New(Expr* expr, QualType eleType) {
  return new (arrayTypePool.Alloc())
         ArrayType(&arrayTypePool, expr, eleType);
}

TileType* TileType::New(const ShapeExpr &expr, QualType eleType) {
  return new (tileTypePool.Alloc())
         TileType(&tileTypePool, expr, eleType);
}

TileType* TileType::New(const ShapeInt &shape, QualType eleType) {
  return new (tileTypePool.Alloc())
         TileType(&tileTypePool, shape, eleType);
}

FuncType* FuncType::New(QualType derived,
                        int funcSpec,
                        bool variadic,
                        const ParamList& params) {
  return new (funcTypePool.Alloc())
         FuncType(&funcTypePool, derived, funcSpec, variadic, params);
}


PointerType* PointerType::New(QualType derived) {
  return new (pointerTypePool.Alloc())
         PointerType(&pointerTypePool, derived);
}


StructType* StructType::New(bool isStruct,
                            bool hasTag,
                            Scope* parent) {
  return new (structUnionTypePool.Alloc())
         StructType(&structUnionTypePool, isStruct, hasTag, parent);
}


int ArithmType::Width() const {
  switch (tag_) {
  case T_BOOL: case T_CHAR: case T_UNSIGNED | T_CHAR:
    return 1;
  case T_SHORT: case T_UNSIGNED | T_SHORT:
    return intWidth_ >> 1;
  case T_INT: case T_UNSIGNED: case T_UNSIGNED | T_INT:
    return intWidth_;
  case T_LONG: case T_UNSIGNED | T_LONG:
    return intWidth_ << 1;
  case T_LLONG: case T_UNSIGNED | T_LLONG:
    return intWidth_ << 1;
  case T_FLOAT:
    return intWidth_;
  case T_DOUBLE:
    return intWidth_ << 1;
  case T_LONG | T_DOUBLE:
    return intWidth_ << 1;
  case T_FLOAT | T_COMPLEX:
    return intWidth_ << 1;
  case T_DOUBLE | T_COMPLEX:
    return intWidth_ << 2;
  case T_LONG | T_DOUBLE | T_COMPLEX:
    return intWidth_ << 2;
  default:
    assert(false);
  }

  return intWidth_; // Make compiler happy
}


int ArithmType::Rank() const {
  switch (tag_) {
  case T_BOOL: return 0;
  case T_CHAR: case T_UNSIGNED | T_CHAR: return 1;
  case T_SHORT: case T_UNSIGNED | T_SHORT: return 2;
  case T_INT: case T_UNSIGNED: case T_UNSIGNED | T_INT: return 3;
  case T_LONG: case T_UNSIGNED | T_LONG: return 4;
  case T_LLONG: case T_UNSIGNED | T_LLONG: return 5;
  case T_FLOAT: return 6;
  case T_DOUBLE: return 7;
  case T_LONG | T_DOUBLE: return 8;
  default:
    assert(tag_ & T_COMPLEX);
    Error("complex not supported yet");
  }
  return 0;
}


ArithmType* ArithmType::MaxType(ArithmType* lhs,
                                ArithmType* rhs) {
  if (lhs->IsInteger())
    lhs = ArithmType::IntegerPromote(lhs);
  if (rhs->IsInteger())
    rhs = ArithmType::IntegerPromote(rhs);
  auto ret = lhs->Rank() > rhs->Rank() ? lhs: rhs;
  if (lhs->Width() == rhs->Width() && (lhs->IsUnsigned() || rhs->IsUnsigned()))
    return ArithmType::New(T_UNSIGNED | ret->Tag());
  return ret;
}


/*
 * Converting from type specifier to type tag
 */
int ArithmType::Spec2Tag(int spec) {
  if (spec == T_SIGNED) {
    return T_INT;
  }
  spec &= ~T_SIGNED;
  if ((spec & T_SHORT) || (spec & T_LONG)
      || (spec & T_LLONG)) {
    spec &= ~T_INT;
  }
  return spec;
}


std::string ArithmType::Str() const {
  std::string width = ":" + std::to_string(Width());

  switch (tag_) {
  case T_BOOL:
    return "bool" + width;

  case T_CHAR:
    return "char" + width;

  case T_UNSIGNED | T_CHAR:
    return "unsigned char" + width;

  case T_SHORT:
    return "short" + width;

  case T_UNSIGNED | T_SHORT:
    return "unsigned short" + width;

  case T_INT:
    return "int" + width;

  case T_UNSIGNED:
    return "unsigned int" + width;

  case T_LONG:
    return "long" + width;

  case T_UNSIGNED | T_LONG:
    return "unsigned long" + width;

  case T_LLONG:
    return "long long" + width;

  case T_UNSIGNED | T_LLONG:
    return "unsigned long long" + width;

  case T_FLOAT:
    return "float" + width;

  case T_DOUBLE:
    return "double" + width;

  case T_LONG | T_DOUBLE:
    return "long double" + width;

  case T_FLOAT | T_COMPLEX:
    return "float complex" + width;

  case T_DOUBLE | T_COMPLEX:
    return "double complex" + width;

  case T_LONG | T_DOUBLE | T_COMPLEX:
    return "long double complex" + width;

  default:
    assert(false);
  }

  return "error"; // Make compiler happy
}


bool PointerType::Compatible(const Type& other) const {
  // C11 6.7.6.1 [2]: pointer compatibility
  auto otherPointer = other.ToPointer();
  return otherPointer && derived_->Compatible(*otherPointer->derived_);

  // FIXME(wgtdkp): cannot loose compatible constraints
  //return other.IsInteger() ||
  //       (otherPointer && derived_->Compatible(*otherPointer->derived_));
}


bool ArrayType::Compatible(const Type& other) const {
  // C11 6.7.6.2 [6]: For two array type to be compatible,
  // the element types must be compatible, and have same length
  // if both specified.
  auto otherArray = other.ToArray();
  if (!otherArray) return false;
  if (!derived_->Compatible(*otherArray->derived_)) return false;
  // The lengths should equal if both specified
  if (complete_ && otherArray->complete_)
    return len_ == otherArray->len_;
  return true;
}

bool TileType::Compatible(const Type& other) const {
  // For two tile type to be compatible,
  // the element types must be compatible, and have same shape
  // if both specified
  auto otherTile = other.ToTile();
  if(!otherTile) return false;
  if (!derived_->Compatible(*otherTile->derived_)) return false;
  // The shapes should be equal if both specified
  if(complete_ && otherTile->complete_)
    return shape_ == otherTile->shape_;
  return true;
}



bool FuncType::Compatible(const Type& other) const {
  auto otherFunc = other.ToFunc();
  // The other type is not an function type
  if (!otherFunc) return false;
  // TODO(wgtdkp): do we need to check the type of return value when deciding
  // compatibility of two function types ??
  if (!derived_->Compatible(*otherFunc->derived_))
    return false;
  if (params_.size() != otherFunc->params_.size())
    return false;

  auto thisIter = params_.begin();
  auto otherIter = otherFunc->params_.begin();
  while (thisIter != params_.end()) {
    if (!(*thisIter)->Type()->Compatible(*(*otherIter)->Type()))
      return false;
    ++thisIter;
    ++otherIter;
  }

  return true;
}


std::string FuncType::Str() const {
  auto str = derived_->Str() + "(";
  auto iter = params_.begin();
  for (; iter != params_.end(); ++iter) {
    str += (*iter)->Type()->Str() + ", ";
  }
  if (variadic_)
    str += "...";
  else if (params_.size())
    str.resize(str.size() - 2);

  return str + ")";
}


StructType::StructType(MemPool* pool,
                       bool isStruct,
                       bool hasTag,
                       Scope* parent)
    : Type(pool, false),
      isStruct_(isStruct),
      hasTag_(hasTag),
      memberMap_(new Scope(parent, S_BLOCK)),
      offset_(0),
      width_(0),
      // If a struct type has no member, it gets alignment of 1
      align_(1),
      bitFieldAlign_(1) {}


Object* StructType::GetMember(const std::string& member) {
  auto ident = memberMap_->FindInCurScope(member);
  if (ident == nullptr)
    return nullptr;
  return ident->ToObject();
}


void StructType::CalcWidth() {
  width_ = 0;
  auto iter = memberMap_->identMap_.begin();
  for (; iter != memberMap_->identMap_.end(); ++iter) {
    width_ += iter->second->Type()->Width();
  }
}


bool StructType::Compatible(const Type& other) const {
  return this == &other; // Pointer comparison
}


// TODO(wgtdkp): more detailed representation
std::string StructType::Str() const {
  std::string str = isStruct_ ? "struct": "union";
  return str + ":" + std::to_string(width_);
}


// Remove useless unnamed bitfield members as they are just for parsing
void StructType::Finalize() {
  for (auto iter = members_.begin(); iter != members_.end();) {
    if ((*iter)->BitFieldWidth() && (*iter)->Anonymous()) {
      members_.erase(iter++);
    } else {
      ++iter;
    }
  }
}


void StructType::AddMember(Object* member) {
  auto offset = MakeAlign(offset_, member->Align());
  member->SetOffset(offset);

  members_.push_back(member);
  memberMap_->Insert(member->Name(), member);

  align_ = std::max(align_, member->Align());
  bitFieldAlign_ = std::max(bitFieldAlign_, align_);

  if (isStruct_) {
    offset_ = offset + member->Type()->Width();
    width_ = MakeAlign(offset_, align_);
  } else {
    assert(offset_ == 0);
    width_ = std::max(width_, member->Type()->Width());
    width_ = MakeAlign(width_, align_);
  }
}


void StructType::AddBitField(Object* bitField, int offset) {
  bitField->SetOffset(offset);
  members_.push_back(bitField);
  if (!bitField->Anonymous())
    memberMap_->Insert(bitField->Name(), bitField);

  auto bytes = MakeAlign(bitField->BitFieldEnd(), 8) / 8;
  bitFieldAlign_ = std::max(bitFieldAlign_, bitField->Align());
  // Does not aligned, default is 1
  if (isStruct_) {
    offset_ = offset + bytes;
    width_ = MakeAlign(offset_, std::max(bitFieldAlign_, bitField->Align()));
  } else {
    assert(offset_ == 0);
    width_ = std::max(width_, bitField->Type()->Width());
  }
}


// Move members of Anonymous struct/union to external struct/union
void StructType::MergeAnony(Object* anony) {
  auto anonyType = anony->Type()->ToStruct();
  auto offset = MakeAlign(offset_, anony->Align());

  // Members in map are never anonymous
  for (auto& kv: *anonyType->memberMap_) {
    auto& name = kv.first;
    auto member = kv.second->ToObject();
    if (member == nullptr) {
      continue;
    }
    // Every member of anonymous struct/union
    // are offseted by external struct/union
    member->SetOffset(offset + member->Offset());

    if (GetMember(name)) {
      Error(member, "duplicated member '%s'", name.c_str());
    }
    // Simplify anony struct's member searching
    memberMap_->Insert(name, member);
  }
  anony->SetOffset(offset);
  members_.push_back(anony);

  align_ = std::max(align_, anony->Align());
  if (isStruct_) {
    offset_ = offset + anonyType->Width();
    width_ = MakeAlign(offset_, align_);
  } else {
    assert(offset_ == 0);
    width_ = std::max(width_, anonyType->Width());
  }
}
