#ifndef _WGTCC_CODE_GEN_H_
#define _WGTCC_CODE_GEN_H_

#include "ast.h"
#include "visitor.h"

namespace triton{
namespace ir{

class value;
class module;

}
}

using namespace triton;

class Parser;
struct Addr;
template<> class Evaluator<Addr>;
struct StaticInitializer;

using TypeList = std::vector<Type*>;
using LocationList = std::vector<std::string>;
using StaticInitList = std::vector<StaticInitializer>;


class Generator: public Visitor {
  friend class Evaluator<Addr>;
public:
  Generator(Parser* parser, ir::module& mod) : parser_(parser), mod_(mod){}

  virtual void Visit(ASTNode* node) { node->Accept(this); }
  void VisitExpr(Expr* expr) { expr->Accept(this); }
  void VisitStmt(Stmt* stmt) { stmt->Accept(this); }

  // Expression
  virtual void VisitBinaryOp(BinaryOp* binaryOp);
  virtual void VisitUnaryOp(UnaryOp* unaryOp);
  virtual void VisitConditionalOp(ConditionalOp* condOp);
  virtual void VisitFuncCall(FuncCall* funcCall);
  virtual void VisitObject(Object* obj);
  virtual void VisitEnumerator(Enumerator* enumer);
  virtual void VisitIdentifier(Identifier* ident);
  virtual void VisitConstant(Constant* cons);
  virtual void VisitTempVar(TempVar* tempVar);

  // Statement
  virtual void VisitDeclaration(Declaration* init);
  virtual void VisitEmptyStmt(EmptyStmt* emptyStmt);
  virtual void VisitIfStmt(IfStmt* ifStmt);
  virtual void VisitJumpStmt(JumpStmt* jumpStmt);
  virtual void VisitReturnStmt(ReturnStmt* returnStmt);
  virtual void VisitLabelStmt(LabelStmt* labelStmt);
  virtual void VisitCompoundStmt(CompoundStmt* compoundStmt);

  virtual void VisitFuncDef(FuncDef* funcDef);
  virtual void VisitTranslationUnit(TranslationUnit* unit);

  void Gen();

protected:
  // Binary
  void GenCommaOp(BinaryOp* comma);
  void GenMemberRefOp(BinaryOp* binaryOp);
  void GenAndOp(BinaryOp* binaryOp);
  void GenOrOp(BinaryOp* binaryOp);
  void GenAddOp(BinaryOp* binaryOp);
  void GenSubOp(BinaryOp* binaryOp);
  void GenAssignOp(BinaryOp* assign);
  void GenCastOp(UnaryOp* cast);
  void GenDerefOp(UnaryOp* deref);
  void GenMinusOp(UnaryOp* minus);
  void GenPointerArithm(BinaryOp* binary);
  void GenDivOp(bool flt, bool sign, int width, int op);
  void GenMulOp(int width, bool flt, bool sign);
  void GenCompOp(int width, bool flt, const char* set);
  void GenCompZero(Type* type);

  // Unary
  void GenIncDec(Expr* operand, bool postfix, const std::string& inst);
  StaticInitializer GetStaticInit(InitList::iterator& iter,
                                  InitList::iterator end, int offset);
  void GenStaticDecl(Declaration* decl);
  void GenSaveArea();
  void GenBuiltin(FuncCall* funcCall);

  void AllocObjects(Scope* scope,
      const FuncDef::ParamList& params=FuncDef::ParamList());

protected:
  Parser* parser_;
  ir::module& mod_;
};


class LValGenerator: public Generator {
public:
  LValGenerator(Parser* parser, ir::module& mod): Generator(parser, mod) {}

  // Expression
  virtual void VisitBinaryOp(BinaryOp* binaryOp);
  virtual void VisitUnaryOp(UnaryOp* unaryOp);
  virtual void VisitObject(Object* obj);
  virtual void VisitIdentifier(Identifier* ident);

  virtual void VisitConditionalOp(ConditionalOp* condOp) { assert(false); }
  virtual void VisitFuncCall(FuncCall* funcCall) { assert(false); }
  virtual void VisitEnumerator(Enumerator* enumer) { assert(false); }
  virtual void VisitConstant(Constant* cons) { assert(false); }
  virtual void VisitTempVar(TempVar* tempVar);

  ir::value* GenExpr(Expr* expr) {
    expr->Accept(this);
    return addr_;
  }

private:
  ir::value* addr_;
};

#endif
