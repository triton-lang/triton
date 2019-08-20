#include "triton/lang/wgtcc/evaluator.h"

#include "triton/lang/wgtcc/ast.h"
#include "triton/lang/wgtcc/code_gen.h"
#include "triton/lang/wgtcc/token.h"


template<typename T>
void Evaluator<T>::VisitBinaryOp(BinaryOp* binary) {
#define L   Evaluator<T>().Eval(binary->lhs_)
#define R   Evaluator<T>().Eval(binary->rhs_)
#define LL  Evaluator<long>().Eval(binary->lhs_)
#define LR  Evaluator<long>().Eval(binary->rhs_)

  if (binary->Type()->ToPointer()) {
    auto val = Evaluator<Addr>().Eval(binary);
    if (val.label_.size()) {
      Error(binary, "expect constant integer expression");
    }
    val_ = static_cast<T>(val.offset_);
    return;
  }

  switch (binary->op_) {
  case '+': val_ = L + R; break;
  case '-': val_ = L - R; break;
  case '*': val_ = L * R; break;
  case '/': {
    auto l = L, r = R;
    if (r == 0)
      Error(binary, "division by zero");
    val_ = l / r;
  } break;
  case '%': {
    auto l = LL, r = LR;
    if (r == 0)
      Error(binary, "division by zero");
    val_ = l % r;
  } break;
  // Bitwise operators that do not accept float
  case '|': val_ = LL | LR; break;
  case '&': val_ = LL & LR; break;
  case '^': val_ = LL ^ LR; break;
  case Token::LEFT: val_ = LL << LR; break;
  case Token::RIGHT: val_ = LL >> LR; break;

  case '<': val_ = L < R; break;
  case '>': val_ = L > R; break;
  case Token::LOGICAL_AND: val_ = L && R; break;
  case Token::LOGICAL_OR: val_ = L || R; break;
  case Token::EQ: val_ = L == R; break;
  case Token::NE: val_ = L != R; break;
  case Token::LE: val_ = L <= R; break;
  case Token::GE: val_ = L >= R; break;
  case '=': case ',': val_ = R; break;
  case '.': {
    auto addr = Evaluator<Addr>().Eval(binary);
    if (addr.label_.size())
      Error(binary, "expect constant expression");
    val_ = addr.offset_;
  }
  default: assert(false);
  }

#undef L
#undef R
#undef LL
#undef LR
}


template<typename T>
void Evaluator<T>::VisitUnaryOp(UnaryOp* unary) {
#define VAL     Evaluator<T>().Eval(unary->operand_)
#define LVAL    Evaluator<long>().Eval(unary->operand_)

  switch (unary->op_) {
  case Token::PLUS: val_ = VAL; break;
  case Token::MINUS: val_ = -VAL; break;
  case '~': val_ = ~LVAL; break;
  case '!': val_ = !VAL; break;
  case Token::CAST:
    if (unary->Type()->IsInteger())
      val_ = static_cast<long>(VAL);
    else
      val_ = VAL;
    break;
  case Token::ADDR: {
    auto addr = Evaluator<Addr>().Eval(unary->operand_);
    if (addr.label_.size())
      Error(unary, "expect constant expression");
    val_ = addr.offset_;
  } break;
  default: Error(unary, "expect constant expression");
  }

#undef LVAL
#undef VAL
}


template<typename T>
void Evaluator<T>::VisitConditionalOp(ConditionalOp* condOp) {
  bool cond;
  auto condType = condOp->cond_->Type();
  if (condType->IsInteger()) {
    auto val = Evaluator<long>().Eval(condOp->cond_);
    cond = val != 0;
  } else if (condType->IsFloat()) {
    auto val = Evaluator<double>().Eval(condOp->cond_);
    cond  = val != 0.0;
  } else if (condType->ToPointer()) {
    auto val = Evaluator<Addr>().Eval(condOp->cond_);
    cond = val.label_.size() || val.offset_;
  } else {
    assert(false);
  }

  if (cond) {
    val_ = Evaluator<T>().Eval(condOp->exprTrue_);
  } else {
    val_ = Evaluator<T>().Eval(condOp->exprFalse_);
  }
}


void Evaluator<Addr>::VisitBinaryOp(BinaryOp* binary) {
#define LR   Evaluator<long>().Eval(binary->rhs_)
#define R   Evaluator<Addr>().Eval(binary->rhs_)

  auto l = Evaluator<Addr>().Eval(binary->lhs_);

  int width = 1;
  auto pointerType = binary->Type()->ToPointer();
  if (pointerType)
    width = pointerType->Derived()->Width();

  switch (binary->op_) {
  case '+':
    assert(pointerType);
    addr_.label_ = l.label_;
    addr_.offset_ = l.offset_ + LR * width;
    break;
  case '-':
    assert(pointerType);
    addr_.label_ = l.label_;
    addr_.offset_ = l.offset_ + LR * width;
    break;
  case '.': {
    addr_.label_ = l.label_;
    auto type = binary->lhs_->Type()->ToStruct();
    auto offset = type->GetMember(binary->rhs_->tok_->str_)->Offset();
    addr_.offset_ = l.offset_ + offset;
    break;
  }
  default: assert(false);
  }
#undef LR
#undef R
}


void Evaluator<Addr>::VisitUnaryOp(UnaryOp* unary) {
  auto addr = Evaluator<Addr>().Eval(unary->operand_);

  switch (unary->op_) {
  case Token::CAST:
  case Token::ADDR:
  case Token::DEREF:
    addr_ = addr; break;
  default: assert(false);
  }
}


void Evaluator<Addr>::VisitConditionalOp(ConditionalOp* condOp) {
  bool cond;
  auto condType = condOp->cond_->Type();
  if (condType->IsInteger()) {
    auto val = Evaluator<long>().Eval(condOp->cond_);
    cond = val != 0;
  } else if (condType->IsFloat()) {
    auto val = Evaluator<double>().Eval(condOp->cond_);
    cond  = val != 0.0;
  } else if (condType->ToPointer()) {
    auto val = Evaluator<Addr>().Eval(condOp->cond_);
    cond = val.label_.size() || val.offset_;
  } else {
    assert(false);
  }

  if (cond) {
    addr_ = Evaluator<Addr>().Eval(condOp->exprTrue_);
  } else {
    addr_ = Evaluator<Addr>().Eval(condOp->exprFalse_);
  }
}


void Evaluator<Addr>::VisitConstant(Constant* cons)  {
  if (cons->Type()->IsInteger()) {
    addr_ = {"", static_cast<int>(cons->IVal())};
  } else if (cons->Type()->ToArray()) {
    Generator().ConsLabel(cons); // Add the literal to rodatas_.
    addr_.label_ = Generator::rodatas_.back().label_;
    addr_.offset_ = 0;
  } else {
    assert(false);
  }
}
