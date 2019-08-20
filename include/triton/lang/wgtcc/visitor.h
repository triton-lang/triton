#ifndef _WGTCC_VISITOR_H_
#define _WGTCC_VISITOR_H_


class BinaryOp;
class UnaryOp;
class ConditionalOp;
class FuncCall;
class Identifier;
class Object;
class Enumerator;
class Constant;
class TempVar;

class Declaration;
class IfStmt;
class JumpStmt;
class ReturnStmt;
class LabelStmt;
class EmptyStmt;
class CompoundStmt;
class FuncDef;
class TranslationUnit;


class Visitor {
public:
  virtual ~Visitor() {}
  virtual void VisitBinaryOp(BinaryOp* binary) = 0;
  virtual void VisitUnaryOp(UnaryOp* unary) = 0;
  virtual void VisitConditionalOp(ConditionalOp* cond) = 0;
  virtual void VisitFuncCall(FuncCall* funcCall) = 0;
  virtual void VisitEnumerator(Enumerator* enumer) = 0;
  virtual void VisitIdentifier(Identifier* ident) = 0;
  virtual void VisitObject(Object* obj) = 0;
  virtual void VisitConstant(Constant* cons) = 0;
  virtual void VisitTempVar(TempVar* tempVar) = 0;

  virtual void VisitDeclaration(Declaration* init) = 0;
  virtual void VisitIfStmt(IfStmt* ifStmt) = 0;
  virtual void VisitJumpStmt(JumpStmt* jumpStmt) = 0;
  virtual void VisitReturnStmt(ReturnStmt* returnStmt) = 0;
  virtual void VisitLabelStmt(LabelStmt* labelStmt) = 0;
  virtual void VisitEmptyStmt(EmptyStmt* emptyStmt) = 0;
  virtual void VisitCompoundStmt(CompoundStmt* compStmt) = 0;
  virtual void VisitFuncDef(FuncDef* funcDef) = 0;
  virtual void VisitTranslationUnit(TranslationUnit* unit) = 0;
};

#endif
