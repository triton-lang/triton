#include "triton/lang/ast.h"
#include "triton/lang/error.h"
#include "triton/lang/evaluator.h"
#include "triton/lang/mem_pool.h"
#include "triton/lang/parser.h"
#include "triton/lang/token.h"


static MemPoolImp<BinaryOp>         binaryOpPool;
static MemPoolImp<TransOp>          transOpPool;
static MemPoolImp<ConditionalOp>    conditionalOpPool;
static MemPoolImp<FuncCall>         funcCallPool;
static MemPoolImp<Declaration>      initializationPool;
static MemPoolImp<Object>           objectPool;
static MemPoolImp<Identifier>       identifierPool;
static MemPoolImp<Enumerator>       enumeratorPool;
static MemPoolImp<Constant>         constantPool;
static MemPoolImp<TempVar>          tempVarPool;
static MemPoolImp<UnaryOp>          unaryOpPool;
static MemPoolImp<EmptyStmt>        emptyStmtPool;
static MemPoolImp<IfStmt>           ifStmtPool;
static MemPoolImp<ForStmt>          forStmtPool;
static MemPoolImp<JumpStmt>         jumpStmtPool;
static MemPoolImp<ReturnStmt>       returnStmtPool;
static MemPoolImp<LabelStmt>        labelStmtPool;
static MemPoolImp<CompoundStmt>     compoundStmtPool;
static MemPoolImp<FuncDef>          funcDefPool;


/*
 * Accept
 */

void Declaration::Accept(Visitor* v) {
  v->VisitDeclaration(this);
}


void EmptyStmt::Accept(Visitor* v) {
  // Nothing to do
}


void LabelStmt::Accept(Visitor* v) {
  v->VisitLabelStmt(this);
}


void IfStmt::Accept(Visitor* v) {
  v->VisitIfStmt(this);
}

void ForStmt::Accept(Visitor* v) {
  v->VisitForStmt(this);
}


void JumpStmt::Accept(Visitor* v) {
  v->VisitJumpStmt(this);
}


void ReturnStmt::Accept(Visitor* v) {
  v->VisitReturnStmt(this);
}


void CompoundStmt::Accept(Visitor* v) {
  v->VisitCompoundStmt(this);
}


void BinaryOp::Accept(Visitor* v) {
  v->VisitBinaryOp(this);
}


void UnaryOp::Accept(Visitor* v) {
  v->VisitUnaryOp(this);
}

void TransOp::Accept(Visitor* v) {
  v->VisitTransOp(this);
}

void ConditionalOp::Accept(Visitor* v) {
  v->VisitConditionalOp(this);
}


void FuncCall::Accept(Visitor* v) {
  v->VisitFuncCall(this);
}


void Identifier::Accept(Visitor* v) {
  v->VisitIdentifier(this);
}


void Object::Accept(Visitor* v) {
  v->VisitObject(this);
}


void Constant::Accept(Visitor* v) {
  v->VisitConstant(this);
}


void Enumerator::Accept(Visitor* v)
{
  v->VisitEnumerator(this);
}


void TempVar::Accept(Visitor* v) {
  v->VisitTempVar(this);
}


void FuncDef::Accept(Visitor* v) {
  v->VisitFuncDef(this);
}


void TranslationUnit::Accept(Visitor* v) {
  v->VisitTranslationUnit(this);
}


// Casting array to pointer, function to pointer to function
Expr* Expr::MayCast(Expr* expr) {
  auto type = Type::MayCast(expr->Type());
  // If the types are equal, no need cast
  if (type != expr->Type()) { // Pointer comparison is enough
    return UnaryOp::New(Token::CAST, expr, type);
  }
  return expr;
}


Expr* Expr::MayCast(Expr* expr, QualType desType) {
  expr = MayCast(expr);
  auto srcType = expr->Type();
  if (desType->ToPointer() && srcType->ToPointer())
    if (desType->IsVoidPointer() || srcType->IsVoidPointer())
      return expr;
  if (!desType->Compatible(*expr->Type()))
    expr = UnaryOp::New(Token::CAST, expr, desType);
  return expr;
}

// Extract the operand's scalar type if possible
// and emit an error otherwise
::Type* Expr::TryExtractScalarType(Expr* loc, Expr *operand) {
  auto scalType = operand->Type()->ScalarType();
  if(!scalType)
    Error(loc, "expect tile or scalar operand");
  return scalType;
}

// If operand is a tile, return a tile of the same shape and
// provided element type
// If operand is a scalar, return provided element type
// directly
::Type* Expr::ScalarOrLikeTile(Expr* operand, ::Type* ty) {
  assert(ty->IsScalar());
  ::Type *retTy = ty;
  if(TileType *T = operand->Type()->ToTile())
    retTy = TileType::New(T->Shape(), retTy);
  return retTy;
}

BinaryOp* BinaryOp::New(const Token* tok, Expr* lhs, Expr* rhs) {
  return New(tok, tok->tag_, lhs, rhs);
}


BinaryOp* BinaryOp::New(const Token* tok, int op, Expr* lhs, Expr* rhs) {
  switch (op) {
  case ',': case '.': case '=':
  case '*': case '/': case '%':
  case '+': case '-': case '&':
  case '^': case '|': case '<':
  case '>':
  case Token::LEFT:
  case Token::RIGHT:
  case Token::LE:
  case Token::GE:
  case Token::EQ:
  case Token::NE:
  case Token::LOGICAL_AND:
  case Token::LOGICAL_OR:
  case Token::ELLIPSIS:
  case Token::MATMUL:
  case Token::MASKED_DEREF:
    break;
  default:
    assert(0);
  }

  auto ret = new (binaryOpPool.Alloc()) BinaryOp(tok, op, lhs, rhs);
  ret->pool_ = &binaryOpPool;

  ret->TypeChecking();
  return ret;
}


ArithmType* BinaryOp::Convert() {
  // Both lhs and rhs are ensured to be have arithmetic scalar type
  auto lhsType = lhs_->Type()->ScalarType()->ToArithm();
  auto rhsType = rhs_->Type()->ScalarType()->ToArithm();
  assert(lhsType && rhsType);
  auto maxType = ArithmType::MaxType(lhsType, rhsType);
  if (lhsType != maxType) { // Pointer comparation is enough!
    lhs_ = UnaryOp::New(Token::CAST, lhs_, ScalarOrLikeTile(lhs_, maxType));
  }
  if (rhsType != maxType) {
    rhs_ = UnaryOp::New(Token::CAST, rhs_, ScalarOrLikeTile(rhs_, maxType));
  }
  return maxType;
}

void BinaryOp::Broadcast(Expr* loc, Expr *&lhs, Expr *&rhs, QualType& type) {
  auto lhsType = lhs->Type()->ToTile();
  auto rhsType = rhs->Type()->ToTile();
  auto eleType = type->ScalarType();
  assert(eleType);
  if(!lhsType && !rhsType)
    return ;
  else if(lhsType && !rhsType){
    type = TileType::New(lhsType->Shape(), eleType);
    ::Type* rtype = TileType::New(lhsType->Shape(), rhs->Type()->ScalarType());
    rhs = UnaryOp::New(Token::CAST, rhs, rtype);
  }
  else if(!lhsType && rhsType){
    type = TileType::New(rhsType->Shape(), eleType);
    ::Type* ltype = TileType::New(rhsType->Shape(), lhs->Type()->ScalarType());
    lhs = UnaryOp::New(Token::CAST, lhs, ltype);

  }
  else {
    auto lhsShape = lhsType->Shape();
    auto rhsShape = rhsType->Shape();
    auto lhsRank = lhsShape.size();
    auto rhsRank = rhsShape.size();
    auto retRank = std::max(lhsRank, rhsRank);
    // pad to the left until shapes have the same rank
    while(lhsShape.size() < retRank)
      lhsShape.insert(lhsShape.begin(), 1);
    while(rhsShape.size() < retRank)
      rhsShape.insert(rhsShape.begin(), 1);
    // broadcast if possible
    TileType::ShapeInt retShape(retRank);
    for(size_t i = 0; i < retRank; i++) {
      if(lhsShape[i] == 1)
        retShape[i] = rhsShape[i];
      else if(rhsShape[i] == 1)
        retShape[i] = lhsShape[i];
      else if(lhsShape[i] == rhsShape[i])
        retShape[i] = lhsShape[i];
      else
        Error(loc, "cannot broadcast dimension %d "
                    "for operands of shape %d and %d",
                    i, lhsShape[i], rhsShape[i]);
    }
    ::Type* ltype = TileType::New(retShape, lhsType->ScalarType());
    ::Type* rtype = TileType::New(retShape, rhsType->ScalarType());
    type = TileType::New(retShape, eleType);
    if(retShape != lhsShape)
      lhs = UnaryOp::New(Token::CAST, lhs, ltype);
    if(retShape != rhsShape)
      rhs = UnaryOp::New(Token::CAST, rhs, rtype);
  }
}

/*
 * Type checking
 */

void Expr::EnsureCompatibleOrVoidPointer(const QualType lhs,
                                         const QualType rhs) const {
  if (lhs->ToPointer() && rhs->ToPointer() &&
      (lhs->IsVoidPointer() || rhs->IsVoidPointer())) {
    return;
  }
  EnsureCompatible(lhs, rhs);
}


void Expr::EnsureCompatible(const QualType lhs, const QualType rhs) const {
  if (!lhs->Compatible(*rhs))
    Error(this, "incompatible types");
}


void BinaryOp::TypeChecking() {
  switch (op_) {
  case '.':
    return MemberRefOpTypeChecking();

  case '*':
  case '/':
  case '%':
    return MultiOpTypeChecking();

  case '+':
  case '-':
    return AdditiveOpTypeChecking();

  case Token::LEFT:
  case Token::RIGHT:
    return ShiftOpTypeChecking();

  case '<':
  case '>':
  case Token::LE:
  case Token::GE:
    return RelationalOpTypeChecking();

  case Token::EQ:
  case Token::NE:
    return EqualityOpTypeChecking();

  case '&':
  case '^':
  case '|':
    return BitwiseOpTypeChecking();

  case Token::LOGICAL_AND:
  case Token::LOGICAL_OR:
    return LogicalOpTypeChecking();

  case '=':
    return AssignOpTypeChecking();

  case ',':
    return CommaOpTypeChecking();

  case Token::ELLIPSIS:
    return RangeOpTypeChecking();

  case Token::MATMUL:
    return MatmulOpTypeChecking();

  case Token::MASKED_DEREF:
    return MaskedDerefOpTypeChecking();

  default:
    assert(0);
  }
}


void BinaryOp::CommaOpTypeChecking() {
  type_ = rhs_->Type();
}


void BinaryOp::SubScriptingOpTypeChecking() {
  assert(false);
}


void BinaryOp::MemberRefOpTypeChecking() {
  type_ = rhs_->Type();
}


void BinaryOp::MultiOpTypeChecking() {
  ::Type* lhsScalType = lhs_->Type()->ScalarType();
  ::Type* rhsScalType = rhs_->Type()->ScalarType();
  if(!lhsScalType || !rhsScalType) {
    Error(this, "operands should have type or scalar type");
  }
  if (!lhsScalType->ToArithm() || !rhsScalType->ToArithm()) {
    Error(this, "operands should have arithmetic type");
  }
  if ('%' == op_ &&
      !(lhsScalType->IsInteger() && rhsScalType->IsInteger())) {
    Error(this, "operands of '%%' should be integers");
  }
  type_ = Convert();
  Broadcast(this, lhs_, rhs_, type_);
}


/*
 * Additive operator is only allowed between:
 *  1. arithmetic types (bool, interger, floating)
 *  2. pointer can be used:
 *    1. lhs of MINUS operator, and rhs must be integer or pointer;
 *    2. lhs/rhs of ADD operator, and the other operand must be integer;
 *  3. tiles can be used:
 *    1. the scalar type of lhs/rhs satisfy the above requirements
 *    2. lhs/rhs that have identical shape
 *    3. lhs/rhs that can be broadcast as per numpy-like semantics
 */
void BinaryOp::AdditiveOpTypeChecking() {
  ::Type* lhsScalType = TryExtractScalarType(this, lhs_);
  ::Type* rhsScalType = TryExtractScalarType(this, rhs_);
  auto lhsPtrType = lhsScalType->ToPointer();
  auto rhsPtrType = rhsScalType->ToPointer();
  if (lhsPtrType) {
    if (op_ == '-') {
      if (rhsPtrType) {
        if (!lhsPtrType->Compatible(*rhsPtrType))
          Error(this, "invalid operands to binary -");
        type_ = ArithmType::New(T_LONG); // ptrdiff_t
      } else if (!rhsScalType->IsInteger()) {
        Error(this, "invalid operands to binary -");
      } else {
        type_ = lhsPtrType;
      }
    } else if (!rhsScalType->IsInteger()) {
      Error(this, "invalid operands to binary +");
    } else {
      type_ = lhsPtrType;
    }
  } else if (rhsPtrType) {
    if (op_ == '+' && !lhsScalType->IsInteger()) {
      Error(this, "invalid operands to binary '+'");
    } else if (op_ == '-' && !lhsPtrType) {
      Error(this, "invalid operands to binary '-'");
    }
    type_ = op_ == '-' ? ArithmType::New(T_LONG): rhsScalType;
    std::swap(lhs_, rhs_); // To simplify code gen
  } else {
    if (!lhsScalType->ToArithm() || !rhsScalType->ToArithm()) {
      Error(this, "invalid operands to binary %s", tok_->str_.c_str());
    }
    type_ = Convert();
  }
  Broadcast(this, lhs_, rhs_, type_);
}

void BinaryOp::RangeOpTypeChecking() {
  auto lhsType = lhs_->Type()->ToArithm();
  auto rhsType = rhs_->Type()->ToArithm();
  if(!lhsType || !lhsType->IsInteger() || !rhsType || !rhsType->IsInteger())
    Error(this, "expect integers for range operator");
  lhs_ = Expr::MayCast(lhs_, ArithmType::IntegerPromote(lhsType));
  rhs_ = Expr::MayCast(rhs_, ArithmType::IntegerPromote(rhsType));
  long begin = Evaluator<long>().Eval(lhs_);
  long end = Evaluator<long>().Eval(rhs_);
  int len = static_cast<int>(end - begin);
  if(len < 0)
    Error(this, "range cannot be negative");
  type_ = TileType::New(TileType::ShapeInt{len}, lhs_->Type());
}

void BinaryOp::MaskedDerefOpTypeChecking() {
//  auto lhsTileType = lhs_->Type()->ToTile();
//  auto rhsTileType = rhs_->Type()->ToTile();
  ::Type* lhsScalType = TryExtractScalarType(this, lhs_);
  ::Type* rhsScalType = TryExtractScalarType(this, rhs_);
  auto lhsType = lhsScalType->ToArithm();
  auto rhsType = rhsScalType->ToPointer();
  if (!rhsType)
    Error(this, "pointer expected for deref pointer in operator '*?'");
  if (!lhsType || (lhsType && !lhsType->IsBool()))
    Error(this, "bool expected for deref mask in operator '*?'");
  type_ = ScalarOrLikeTile(rhs_, rhsType->Derived().GetPtr());
  Broadcast(this, lhs_, rhs_, type_);
}

void BinaryOp::MatmulOpTypeChecking() {
  auto lhsType = lhs_->Type()->ToTile();
  auto rhsType = rhs_->Type()->ToTile();
  if(!lhsType || !rhsType)
    Error(this, "expect tile operands for matrix multiplication");
  auto lhsShape = lhsType->Shape();
  auto rhsShape = rhsType->Shape();
  size_t lhsRank = lhsShape.size();
  size_t rhsRank = rhsShape.size();
  if(lhsRank != rhsRank)
    Error(this, "matrix multiplication operands have incompatible rank"
                "%d and %d", lhsRank, rhsRank);
  for(int d = 2; d < lhsRank; d++)
    if(lhsShape[d] != rhsShape[d])
      Error(this, "matrix multiplication operands have incompatible batch dimension"
                  "%d and %d for axis %d", lhsShape[d], rhsShape[d], d);
  if(lhsShape[1] != rhsShape[0])
    Error(this, "matrix multiplication operands have incompatible inner dimension"
                " %d and %d", lhsShape[1], rhsShape[0]);
  // ret shape
  TileType::ShapeInt retShape = {lhsShape[0], rhsShape[1]};
  for(int d = 2; d < lhsRank; d++)
    retShape.push_back(lhsShape[d]);
  QualType retType = lhsType->Derived();
  if(retType != rhsType->Derived())
    Error(this, "matrix multiplication operands have incompatible data types");
  ArithmType* ScalType = lhsType->ScalarType()->ToArithm();
  if(ScalType->Tag() & T_HALF)
    ScalType = ArithmType::New(T_FLOAT);
  type_ = TileType::New(retShape, ScalType);
}

void BinaryOp::ShiftOpTypeChecking() {
  ::Type* lhsScalType = TryExtractScalarType(this, lhs_);
  ::Type* rhsScalType = TryExtractScalarType(this, rhs_);
  auto lhsType = lhsScalType->ToArithm();
  auto rhsType = rhsScalType->ToArithm();
  if (!lhsType || !lhsType->IsInteger() || !rhsType || !rhsType->IsInteger())
    Error(this, "expect integers for shift operator");
  lhs_ = Expr::MayCast(lhs_, ScalarOrLikeTile(lhs_, ArithmType::IntegerPromote(lhsType)));
  rhs_ = Expr::MayCast(rhs_, ScalarOrLikeTile(rhs_, ArithmType::IntegerPromote(rhsType)));
  type_ = lhs_->Type();
  Broadcast(this, lhs_, rhs_, type_);
}


void BinaryOp::RelationalOpTypeChecking() {
  ::Type* lhsScalType = TryExtractScalarType(this, lhs_);
  ::Type* rhsScalType = TryExtractScalarType(this, rhs_);
  if (lhsScalType->ToPointer() || rhsScalType->ToPointer()) {
    EnsureCompatible(lhsScalType, rhsScalType);
  } else {
    if (!lhsScalType->IsReal() || !rhsScalType->IsReal()) {
      Error(this, "expect real type of operands");
    }
    Convert();
  }
  type_ = ArithmType::New(T_INT);
  Broadcast(this, lhs_, rhs_, type_);
}


void BinaryOp::EqualityOpTypeChecking() {
  ::Type* lhsScalType = TryExtractScalarType(this, lhs_);
  ::Type* rhsScalType = TryExtractScalarType(this, rhs_);
  if (lhsScalType->ToPointer() || rhsScalType->ToPointer()) {
    EnsureCompatibleOrVoidPointer(lhsScalType, rhsScalType);
  } else {
    if (!lhsScalType->ToArithm() || !rhsScalType->ToArithm())
      Error(this, "invalid operands to binary %s", tok_->str_.c_str());
    Convert();
  }
  type_ = ArithmType::New(T_INT);
  Broadcast(this, lhs_, rhs_, type_);
}


void BinaryOp::BitwiseOpTypeChecking() {
  ::Type* lhsScalType = TryExtractScalarType(this, lhs_);
  ::Type* rhsScalType = TryExtractScalarType(this, rhs_);
  if (!lhsScalType->IsInteger() || !rhsScalType->IsInteger())
    Error(this, "operands of '&' should be integer");
  type_ = Convert();
  Broadcast(this, lhs_, rhs_, type_);
}


void BinaryOp::LogicalOpTypeChecking() {
  ::Type* lhsScalType = TryExtractScalarType(this, lhs_);
  ::Type* rhsScalType = TryExtractScalarType(this, rhs_);
  if (!lhsScalType->IsScalar() || !rhsScalType->IsScalar())
    Error(this, "the operand should be arithmetic type or pointer");
  type_ = ArithmType::New(T_INT);
  Broadcast(this, lhs_, rhs_, type_);
}


void BinaryOp::AssignOpTypeChecking() {
  if (lhs_->IsConstQualified()) {
    Error(lhs_, "left operand of '=' is const qualified");
  } else if (!lhs_->IsLVal()) {
    Error(lhs_, "lvalue expression expected");
  }

  ::Type* lhsScalType = TryExtractScalarType(this, lhs_);
  ::Type* rhsScalType = TryExtractScalarType(this, rhs_);
  if (!lhsScalType->ToArithm() || !rhsScalType->ToArithm()) {
    EnsureCompatibleOrVoidPointer(lhsScalType, rhsScalType);
  }

  // The other constraints are lefted to cast operator
  rhs_ = Expr::MayCast(rhs_, ScalarOrLikeTile(rhs_, lhsScalType));
  type_ = lhs_->Type();
  rhs_ = UnaryOp::New(Token::CAST, rhs_, type_);
}

/*
 * Unary Operators
 */

UnaryOp* UnaryOp::New(int op, Expr* operand, QualType type, int info) {
  auto ret = new (unaryOpPool.Alloc()) UnaryOp(op, operand, type, info);
  ret->pool_ = &unaryOpPool;

  ret->TypeChecking();
  return ret;
}


int UnaryOp::encodeRed(int ax, int tag) {
  int result = 0;
  result |= ax;
  result |= tag << 16;
  return result;
}

void UnaryOp::decodeRed(int info, int& ax, int& tag) {
  ax = info & 0x0000FFFF;
  tag = (info & 0xFFFF0000) >> 16;
}

bool UnaryOp::IsLVal() {
  // Only deref('*') could be lvalue;
  return op_ == Token::DEREF;
}


::Type* UnaryOp::Convert() {
  auto scalType = operand_->Type()->ScalarType();
  assert(scalType);
  auto arithmType = scalType->ToArithm();
  assert(arithmType);
  if (arithmType->IsInteger())
    arithmType = ArithmType::IntegerPromote(arithmType);
  ::Type* retType = ScalarOrLikeTile(operand_, arithmType);
  operand_ = Expr::MayCast(operand_, retType);
  return retType;
}


void UnaryOp::TypeChecking() {
  switch (op_) {
  case Token::POSTFIX_INC:
  case Token::POSTFIX_DEC:
  case Token::PREFIX_INC:
  case Token::PREFIX_DEC:
    return IncDecOpTypeChecking();

  case Token::ADDR:
    return AddrOpTypeChecking();

  case Token::DEREF:
    return DerefOpTypeChecking();

  case Token::PLUS:
  case Token::MINUS:
  case '~':
  case '!':
    return UnaryArithmOpTypeChecking();

  case Token::BITCAST:
    return BitcastOpTypeChecking();

  case Token::CAST:
    return CastOpTypeChecking();

  case Token::REDUCE:
    return ReduceOpTypeChecking();

  case Token::EXP:
    return IntrinsicOpTypeChecking();

  default:
    assert(false);
  }
}

void UnaryOp::IncDecOpTypeChecking() {
  if (operand_->IsConstQualified()) {
    Error(this, "increment/decrement of const qualified expression");
  } else if (!operand_->IsLVal()) {
    Error(this, "lvalue expression expected");
  }
  auto scalType = TryExtractScalarType(this, operand_);
  if (!scalType->IsReal() && !scalType->ToPointer()) {
    Error(this, "expect operand of real type or pointer");
  }
  type_ = operand_->Type();
}


void UnaryOp::AddrOpTypeChecking() {
  auto funcType = operand_->Type()->ToFunc();
  if (funcType == nullptr && !operand_->IsLVal())
    Error(this, "expression must be an lvalue or function designator");
  if(operand_->Type()->IsTile())
    Error(this, "cannot take the address of a tile");
  type_ = PointerType::New(operand_->Type());
}


void UnaryOp::DerefOpTypeChecking() {
  auto scalType = TryExtractScalarType(this, operand_);
  auto pointerType = scalType->ToPointer();
  if (!pointerType)
    Error(this, "pointer expected for deref operator '*'");
  type_ = ScalarOrLikeTile(operand_, pointerType->Derived().GetPtr());
}

void UnaryOp::ReduceOpTypeChecking() {
  int ax, tag;
  decodeRed(info_, ax, tag);
  auto tileType = operand_->Type()->ToTile();
  if(!tileType)
    Error(this, "array expected for reduction operation");
  auto shape = tileType->Shape();
  shape.erase(shape.begin() + ax);
  if(shape.empty())
    type_ = tileType->Derived();
  else
    type_ = TileType::New(shape, tileType->Derived());
}

void UnaryOp::UnaryArithmOpTypeChecking() {
  auto scalType = TryExtractScalarType(this, operand_);
  if (Token::PLUS == op_ || Token::MINUS == op_) {
    if (!scalType->ToArithm())
      Error(this, "Arithmetic type expected");
    Convert();
    type_ = operand_->Type();
  } else if ('~' == op_) {
    if (!scalType->IsInteger())
      Error(this, "integer expected for operator '~'");
    Convert();
    type_ = operand_->Type();
  } else if (!scalType->IsScalar()) {
    Error(this, "arithmetic type or pointer expected for operator '!'");
  } else {
    type_ = ScalarOrLikeTile(operand_, ArithmType::New(T_INT));
  }
}

void UnaryOp::BitcastOpTypeChecking() {
  auto operandType = Type::MayCast(operand_->Type());
  if(type_->Width() != operandType->Width())
    Error(this, "cannot bitcast to type of different width");
}

void UnaryOp::CastOpTypeChecking() {
  auto operandType = Type::MayCast(operand_->Type());
  // The type_ has been initiated to dest type
  if (type_->ToVoid()) {
    // The expression becomes a void expression
  } else if(type_->IsTile() || operandType->IsTile()) {
    /* Broadcasting rules:
     *   1. Tiles with 1 element can be converted to scalar
     *   2. Scalar can be converted to tiles of any shapes
     *   3. Tiles can be converted to another tile only if the
     *       mismatching dimensions are unitary
     */
    if(type_->IsScalar() && operandType->ToTile()->NumEle() != 1)
      Error(this, "tile with more than one element cannot be casted to scalar");
    if(type_->IsTile() && operandType->IsTile()){
      auto shape = type_->ToTile()->Shape();
      auto operandShape = operandType->ToTile()->Shape();
      if(operandShape.size() > shape.size())
        Error(this, "cast cannot reduce operand rank");
      while(operandShape.size() < shape.size())
        operandShape.insert(operandShape.begin(), 1);
      for(size_t i = 0; i < shape.size(); i++) {
        if(shape[i] != 1 && operandShape[i] != 1 && shape[i] != operandShape[i])
          Error(this, "cannot broadcast dimension %d "
                      "for operands of shape %d and %d",
                      i, shape[i], operandShape[i]);
      }
    }
  } else if (!type_->IsScalar() || !operandType->IsScalar()) {
    if (!type_->Compatible(*operandType))
      Error(this, "the cast type should be arithemetic type or pointer");
  } else if (type_->IsFloat() && operandType->ToPointer()) {
    Error(this, "cannot cast a pointer to floating");
  } else if (type_->ToPointer() && operandType->IsFloat()) {
    Error(this, "cannot cast a floating to pointer");
  }
}

void UnaryOp::IntrinsicOpTypeChecking() {
  type_ = ScalarOrLikeTile(operand_, ArithmType::New(T_FLOAT));
}

/*
 * Transposition Operator
 */
void TransOp::TypeChecking() {
  auto tileType = operand_->Type()->ToTile();
  if(!tileType)
    Error(this, "tile expected for transposition operator '^'");
  auto opShape = tileType->Shape();
  if(perm_.size() != opShape.size())
    Error(this, "invalid permutations");
  // permutate input shape
  TileType::ShapeInt resShape(opShape.size());
  for(int d = 0; d < opShape.size(); d++)
    resShape[d] = opShape[perm_[d]];
  type_ = TileType::New(resShape, tileType->Derived());
}

TransOp* TransOp::New(const PermInt& perm, Expr* operand) {
  auto ret = new (transOpPool.Alloc()) TransOp(perm, operand);
  ret->pool_ = &transOpPool;
  ret->TypeChecking();
  return ret;
}

/*
 * Conditional Operator
 */

ConditionalOp* ConditionalOp::New(const Token* tok,
                                  Expr* cond,
                                  Expr* exprTrue,
                                  Expr* exprFalse) {
  auto ret = new (conditionalOpPool.Alloc())
      ConditionalOp(cond, exprTrue, exprFalse);
  ret->pool_ = &conditionalOpPool;

  ret->TypeChecking();
  return ret;
}


ArithmType* ConditionalOp::Convert() {
  auto lhsType = exprTrue_->Type()->ScalarType()->ToArithm();
  auto rhsType = exprFalse_->Type()->ScalarType()->ToArithm();
  assert(lhsType && rhsType);
  auto type = ArithmType::MaxType(lhsType, rhsType);
  if (lhsType != type) { // Pointer comparation is enough!
    exprTrue_ = UnaryOp::New(Token::CAST, exprTrue_, type);
  }
  if (rhsType != type) {
    exprFalse_ = UnaryOp::New(Token::CAST, exprFalse_, type);
  }

  return type;
}


void ConditionalOp::TypeChecking() {
  auto condScalarType = TryExtractScalarType(this, cond_);

  if (!condScalarType) {
    Error(cond_->Tok(), "condition must be tile or scalar");
  }

  auto lhsType = TryExtractScalarType(this, exprTrue_);
  auto rhsType = TryExtractScalarType(this, exprFalse_);
  if (lhsType->ToArithm() && rhsType->ToArithm()) {
    type_ = Convert();
  } else {
    EnsureCompatibleOrVoidPointer(lhsType, rhsType);
    type_ = lhsType;
  }
  BinaryOp::Broadcast(this, exprFalse_, exprTrue_, type_);
}


/*
 * Function Call
 */

FuncCall* FuncCall::New(Expr* designator, const ArgList& args) {
  auto ret = new (funcCallPool.Alloc()) FuncCall(designator, args);
  ret->pool_ = &funcCallPool;

  ret->TypeChecking();
  return ret;
}


void FuncCall::TypeChecking() {
  auto pointerType = designator_->Type()->ToPointer();
  if (pointerType) {
    if (!pointerType->Derived()->ToFunc())
      Error(designator_, "called object is not a function or function pointer");
    // Convert function pointer to function type
    designator_ = UnaryOp::New(Token::DEREF, designator_);
  }
  auto funcType = designator_->Type()->ToFunc();
  if (!funcType) {
    Error(designator_, "called object is not a function or function pointer");
  } else if (!funcType->Derived()->ToVoid() &&
             !funcType->Derived()->Complete()) {
    Error(designator_, "invalid use of incomplete return type");
  }

  auto arg = args_.begin();
  for (auto param: funcType->Params()) {
    if (arg == args_.end())
      Error(this, "too few arguments for function call");
    *arg = Expr::MayCast(*arg, param->Type());
    ++arg;
  }
  if (arg != args_.end() && !funcType->Variadic())
    Error(this, "too many arguments for function call");

  // C11 6.5.2.2 [6]: promote float to double if it has no prototype
  while (arg != args_.end()) {
    if ((*arg)->Type()->IsFloat() && (*arg)->Type()->Width() == 4) {
      auto type = ArithmType::New(T_DOUBLE);
      *arg = UnaryOp::New(Token::CAST, *arg, type);
    }
    ++arg;
  }

  type_ = funcType->Derived();
}


/*
 * Identifier
 */

Identifier* Identifier::New(const Token* tok,
                            QualType type,
                            enum Linkage linkage,
                            const AttrList &attrList) {
  auto ret = new (identifierPool.Alloc()) Identifier(tok, type, linkage, attrList);
  ret->pool_ = &identifierPool;
  return ret;
}


Enumerator* Enumerator::New(const Token* tok, int val) {
  auto ret = new (enumeratorPool.Alloc()) Enumerator(tok, val);
  ret->pool_ = &enumeratorPool;
  return ret;
}


Declaration* Declaration::New(Object* obj) {
  auto ret = new (initializationPool.Alloc()) Declaration(obj);
  ret->pool_ = &initializationPool;
  return ret;
}

void Declaration::AddInit(Initializer init) {
  init.expr_ = Expr::MayCast(init.expr_, init.type_);
  auto res = inits_.insert(init);
  if (!res.second) {
    inits_.erase(res.first);
    inits_.insert(init);
  }
}


/*
 * Object
 */

Object* Object::New(const Token* tok,
                    QualType type,
                    int storage,
                    enum Linkage linkage,
                    unsigned char bitFieldBegin,
                    unsigned char bitFieldWidth,
                    const AttrList& attrList) {
  auto ret = new (objectPool.Alloc())
             Object(tok, type, storage, linkage, bitFieldBegin, bitFieldWidth, attrList);
  ret->pool_ = &objectPool;

  static long id = 0;
  if (ret->IsStatic() || ret->Anonymous())
    ret->id_ = ++id;
  return ret;
}


Object* Object::NewAnony(const Token* tok,
                         QualType type,
                         int storage,
                         enum Linkage linkage,
                         unsigned char bitFieldBegin,
                         unsigned char bitFieldWidth,
                         const AttrList& attrList) {
  auto ret = new (objectPool.Alloc())
             Object(tok, type, storage, linkage, bitFieldBegin, bitFieldWidth, attrList);
  ret->pool_ = &objectPool;
  ret->anonymous_ = true;

  static long id = 0;
  if (ret->IsStatic() || ret->anonymous_)
    ret->id_ = ++id;
  return ret;
}


/*
 * Constant
 */

Constant* Constant::New(const Token* tok, int tag, long val) {
  auto type = ArithmType::New(tag);
  auto ret = new (constantPool.Alloc()) Constant(tok, type, val);
  ret->pool_ = &constantPool;
  return ret;
}


Constant* Constant::New(const Token* tok, int tag, double val) {
  auto type = ArithmType::New(tag);
  auto ret = new (constantPool.Alloc()) Constant(tok, type, val);
  ret->pool_ = &constantPool;
  return ret;
}


Constant* Constant::New(const Token* tok, int tag, const std::string* val) {
  auto derived = ArithmType::New(tag);
  auto type = ArrayType::New(val->size() / derived->Width(), derived);

  auto ret = new (constantPool.Alloc()) Constant(tok, type, val);
  ret->pool_ = &constantPool;

  static long id = 0;
  ret->id_ = ++id;
  return ret;
}


std::string Constant::SValRepr() const {
  std::vector<char> buf(4 * sval_->size() + 1);
  for (size_t i = 0; i < sval_->size(); ++i) {
    int c = (*sval_)[i];
    sprintf(&buf[i * 4], "\\x%1x%1x", (c >> 4) & 0xf, c & 0xf);
  }
  return std::string(buf.begin(), buf.end() - 1);
}


/*
 * TempVar
 */

TempVar* TempVar::New(QualType type) {
  auto ret = new (tempVarPool.Alloc()) TempVar(type);
  ret->pool_ = &tempVarPool;
  return ret;
}


/*
 * Statement
 */

EmptyStmt* EmptyStmt::New() {
  auto ret = new (emptyStmtPool.Alloc()) EmptyStmt();
  ret->pool_ = &emptyStmtPool;
  return ret;
}


// The else stmt could be null
IfStmt* IfStmt::New(Expr* cond, Stmt* then, Stmt* els) {
  auto ret = new (ifStmtPool.Alloc()) IfStmt(cond, then, els);
  ret->pool_ = &ifStmtPool;
  return ret;
}


CompoundStmt* CompoundStmt::New(std::list<Stmt*>& stmts, ::Scope* scope) {
  auto ret = new (compoundStmtPool.Alloc()) CompoundStmt(stmts, scope);
  ret->pool_ = &compoundStmtPool;
  return ret;
}

ForStmt* ForStmt::New(Stmt* body, Stmt* init, Expr* cond, Expr* step) {
  auto ret = new (forStmtPool.Alloc()) ForStmt(body, init, cond, step);
  ret->pool_ = &forStmtPool;
  return ret;
}

JumpStmt* JumpStmt::New(LabelStmt* label) {
  auto ret = new (jumpStmtPool.Alloc()) JumpStmt(label);
  ret->pool_ = &jumpStmtPool;
  return ret;
}


ReturnStmt* ReturnStmt::New(Expr* expr) {
  auto ret = new (returnStmtPool.Alloc()) ReturnStmt(expr);
  ret->pool_ = &returnStmtPool;
  return ret;
}


LabelStmt* LabelStmt::New() {
  auto ret = new (labelStmtPool.Alloc()) LabelStmt();
  ret->pool_ = &labelStmtPool;
  return ret;
}


FuncDef* FuncDef::New(Identifier* ident, LabelStmt* retLabel) {
  auto ret = new (funcDefPool.Alloc()) FuncDef(ident, retLabel);
  ret->pool_ = &funcDefPool;
  return ret;
}


bool Initializer::operator<(const Initializer& rhs) const {
  if (offset_ < rhs.offset_)
    return true;
  return (offset_ == rhs.offset_ && bitFieldBegin_ < rhs.bitFieldBegin_);
}
