#pragma once

#ifndef _WGTCC_CODE_GEN_H_
#define _WGTCC_CODE_GEN_H_

#include "ast.h"
#include "visitor.h"
#include <stack>

namespace triton{
namespace ir{

class value;
class module;
class type;
class context;
class builder;
class attribute;

}
}

using namespace triton;

class Parser;
struct Addr;
template<> class Evaluator<Addr>;
struct StaticInitializer;
class LValAssigner;

using TypeList = std::vector<Type*>;
using LocationList = std::vector<std::string>;
using StaticInitList = std::vector<StaticInitializer>;

// Error
inline void should_not_happen() { throw std::runtime_error("should not happen"); }
inline void error_not_implemented() { throw std::runtime_error("not implemented"); }

class Generator: public Visitor {
  friend class Evaluator<Addr>;
  friend class LValAssigner;

protected:
  struct scope {
    std::map<std::string, ir::type*> types;
    std::map<std::string, ir::value*> values;
  };

  void set_ret(ir::value* value);
  ir::value *GenUnaryMinus(ir::value* arg);

public:
  Generator(Parser* parser) : parser_(parser) {}

  void Visit(ASTNode* node) { node->Accept(this); }
  void VisitExpr(Expr* expr) { expr->Accept(this); }
  void VisitStmt(Stmt* stmt) { stmt->Accept(this); }

  // Expression
  void VisitBinaryOp(BinaryOp* binaryOp);
  void VisitUnaryOp(UnaryOp* unaryOp);
  void VisitTransOp(TransOp* transOp);
  void VisitConditionalOp(ConditionalOp* condOp);
  void VisitFuncCall(FuncCall* funcCall);
  void VisitObject(Object* obj);
  void VisitEnumerator(Enumerator* enumer);
  void VisitIdentifier(Identifier* ident);
  void VisitConstant(Constant* cons);
  void VisitTempVar(TempVar* tempVar);

  // Statement
  void VisitDeclaration(Declaration* init);
  void VisitEmptyStmt(EmptyStmt* emptyStmt);
  void VisitIfStmt(IfStmt* ifStmt);
  void VisitForStmt(ForStmt* ifStmt);
  void VisitJumpStmt(JumpStmt* jumpStmt);
  void VisitReturnStmt(ReturnStmt* returnStmt);
  void VisitLabelStmt(LabelStmt* labelStmt);
  void VisitCompoundStmt(CompoundStmt* compoundStmt);

  void VisitFuncDef(FuncDef* funcDef);
  void VisitTranslationUnit(TranslationUnit* unit);

  void Gen(ir::module *mod);

protected:
  // Triton-IR attributes
  ir::attribute GenIRAttr(ASTNode::Attr attr);

  // Triton-IR values
  ir::value* GenAssignOp(Expr* lvalue, ir::value* rhs);
  ir::value* GenBroadcastOp(ir::value* src, ir::type* dst_ty);
  ir::value* GenNumcastOp(ir::value*src, ir::type* dst_ty);
  ir::value* GenCastOp(ir::value* op, ir::type* type);

  // Triton-IR types
  static ir::type* GenIRType(::Type* type, ir::context &ctx);
  static ir::type* GenIRArithmType(ArithmType* type, ir::context& ctx);
  static ir::type* GenIRArrayType(ArrayType* type,  ir::context& ctx);
  static ir::type* GenIRTileType(TileType* type,  ir::context& ctx);
  static ir::type* GenIRFuncType(FuncType* type,  ir::context& ctx);
  static ir::type* GenIRPointerType(PointerType* type,  ir::context& ctx);
  static ir::type* GenIRStructType(StructType* type,  ir::context& ctx);
  void AllocObjects(Scope* scope, const FuncDef::ParamList& params=FuncDef::ParamList());

  // SSA
  void pushScope();
  void popScope();

private:
  Parser* parser_;
  ir::value* ret_;
  ir::builder* bld_;
  ir::context* ctx_;
  ir::module* mod_;

private:
//  std::stack<scope> scopes_;
  LValAssigner* assign_;
};


class LValAssigner: public Visitor {
public:
  LValAssigner(Generator* gen): gen_(gen) {}

  // Expression
  void VisitBinaryOp(BinaryOp* binaryOp);
  void VisitUnaryOp(UnaryOp* unaryOp);
  void VisitObject(Object* obj);
  void VisitIdentifier(Identifier* ident);

  void VisitConditionalOp(ConditionalOp*)      { should_not_happen(); }
  void VisitFuncCall(FuncCall*)                { should_not_happen(); }
  void VisitTransOp(TransOp*)                  { should_not_happen(); }
  void VisitEnumerator(Enumerator*)            { should_not_happen(); }
  void VisitConstant(Constant*)                { should_not_happen(); }
  void VisitTempVar(TempVar*)                  { should_not_happen(); }
  void VisitDeclaration(Declaration*)          { should_not_happen(); }
  void VisitEmptyStmt(EmptyStmt*)              { should_not_happen(); }
  void VisitIfStmt(IfStmt*)                    { should_not_happen(); }
  void VisitForStmt(ForStmt*)                  { should_not_happen(); }
  void VisitJumpStmt(JumpStmt*)                { should_not_happen(); }
  void VisitReturnStmt(ReturnStmt*)            { should_not_happen(); }
  void VisitLabelStmt(LabelStmt*)              { should_not_happen(); }
  void VisitCompoundStmt(CompoundStmt*)        { should_not_happen(); }
  void VisitFuncDef(FuncDef*)                  { should_not_happen(); }
  void VisitTranslationUnit(TranslationUnit*)  { should_not_happen(); }

  ir::value* GenExpr(Expr* expr, ir::value* rhs) {
    rhs_ = rhs;
    expr->Accept(this);
    return ret_;
  }

private:
  ir::value* ret_;
  ir::value* rhs_;
  Generator* gen_;
};

#endif
