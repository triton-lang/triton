#pragma once

#ifndef _WGTCC_EVALUATOR_H_
#define _WGTCC_EVALUATOR_H_

#include "ast.h"
#include "error.h"
#include "visitor.h"


class Expr;

template<typename T>
class Evaluator: public Visitor {
public:
  Evaluator() {}

  virtual ~Evaluator() {}

  virtual void VisitBinaryOp(BinaryOp* binary);
  virtual void VisitUnaryOp(UnaryOp* unary);
  virtual void VisitConditionalOp(ConditionalOp* cond);

  virtual void VisitFuncCall(FuncCall* funcCall) {
    Error(funcCall, "expect constant expression");
  }
  virtual void VisitEnumerator(Enumerator* enumer) {
    val_ = static_cast<T>(enumer->Val());
  }
  virtual void VisitIdentifier(Identifier* ident) {
    Error(ident, "expect constant expression");
  }
  virtual void VisitObject(Object* obj) {
    Error(obj, "expect constant expression");
  }
  virtual void VisitConstant(Constant* cons) {
    if (cons->Type()->IsFloat()) {
      val_ = static_cast<T>(cons->FVal());
    } else if (cons->Type()->IsInteger()) {
      val_ = static_cast<T>(cons->IVal());
    } else {
      assert(false);
    }
  }
  virtual void VisitTempVar(TempVar* tempVar) { assert(false); }

  // We may should assert here
  virtual void VisitDeclaration(Declaration* init) {}
  virtual void VisitIfStmt(IfStmt* ifStmt) {}
  virtual void VisitForStmt(ForStmt* forStmt) {}
  virtual void VisitJumpStmt(JumpStmt* jumpStmt) {}
  virtual void VisitReturnStmt(ReturnStmt* returnStmt) {}
  virtual void VisitLabelStmt(LabelStmt* labelStmt) {}
  virtual void VisitEmptyStmt(EmptyStmt* emptyStmt) {}
  virtual void VisitCompoundStmt(CompoundStmt* compStmt) {}
  virtual void VisitFuncDef(FuncDef* funcDef) {}
  virtual void VisitTranslationUnit(TranslationUnit* unit) {}

  T Eval(Expr* expr) {
    expr->Accept(this);
    return val_;
  }

private:
  T val_;
};


struct Addr {
  std::string label_;
  int offset_;
};

template<>
class Evaluator<Addr>: public Visitor {
public:
  Evaluator<Addr>() {}
  virtual ~Evaluator<Addr>() {}
  virtual void VisitBinaryOp(BinaryOp* binary);
  virtual void VisitUnaryOp(UnaryOp* unary);
  virtual void VisitConditionalOp(ConditionalOp* cond);

  virtual void VisitFuncCall(FuncCall* funcCall) {
    Error(funcCall, "expect constant expression");
  }
  virtual void VisitEnumerator(Enumerator* enumer) {
    addr_.offset_ = enumer->Val();
  }
  virtual void VisitIdentifier(Identifier* ident) {
    addr_.label_ = ident->Name();
    addr_.offset_ = 0;
  }
  virtual void VisitObject(Object* obj) {
    if (!obj->IsStatic()) {
      Error(obj, "expect static object");
    }
    addr_.label_ = obj->Repr();
    addr_.offset_ = 0;
  }
  virtual void VisitConstant(Constant* cons);
  virtual void VisitTempVar(TempVar* tempVar) { assert(false); }

  // We may should assert here
  virtual void VisitDeclaration(Declaration* init) {}
  virtual void VisitIfStmt(IfStmt* ifStmt) {}
  virtual void VisitForStmt(ForStmt* forStmt) {}
  virtual void VisitJumpStmt(JumpStmt* jumpStmt) {}
  virtual void VisitReturnStmt(ReturnStmt* returnStmt) {}
  virtual void VisitLabelStmt(LabelStmt* labelStmt) {}
  virtual void VisitEmptyStmt(EmptyStmt* emptyStmt) {}
  virtual void VisitCompoundStmt(CompoundStmt* compStmt) {}
  virtual void VisitFuncDef(FuncDef* funcDef) {}
  virtual void VisitTranslationUnit(TranslationUnit* unit) {}

  Addr Eval(Expr* expr) {
    expr->Accept(this);
    return addr_;
  }

private:
  Addr addr_;
};

#endif
