#pragma once

#ifndef _PARSER_H_
#define _PARSER_H_

#include "ast.h"
#include "encoding.h"
#include "error.h"
#include "mem_pool.h"
#include "scope.h"
#include "token.h"

#include <cassert>
#include <memory>
#include <stack>


class Preprocessor;

struct DeclInfo {
  DeclInfo(const Token* _tok,
           QualType _type,
           ASTNode::AttrList _attrs = {})
    : tok(_tok), type(_type), attrs(_attrs) {}

  const Token* tok;
  QualType type;
  ASTNode::AttrList attrs;
};


class Parser {
  using LiteralList = std::vector<Constant*>;
  using StaticObjectList = std::vector<Object*>;
  using CaseLabelList = std::vector<std::pair<Constant*, LabelStmt*>>;
  using LabelJumpList = std::list<std::pair<const Token*, JumpStmt*>>;
  using LabelMap = std::map<std::string, LabelStmt*>;
  friend class Generator;

public:
  explicit Parser(TokenSequence& ts)
    : unit_(TranslationUnit::New()),
      ts_(ts),
      externalSymbols_(new Scope(nullptr, S_BLOCK)),
      errTok_(nullptr),
      curScope_(new Scope(nullptr, S_FILE)),
      curFunc_(nullptr),
      breakDest_(nullptr),
      continueDest_(nullptr),
      caseLabels_(nullptr),
      defaultLabel_(nullptr) {
        ts_.SetParser(this);
      }

  ~Parser() {}

  Constant* ParseConstant(const Token* tok);
  Constant* ParseFloat(const Token* tok);
  Constant* ParseInteger(const Token* tok);
  Constant* ParseCharacter(const Token* tok);
  Encoding ParseLiteral(std::string& str, const Token* tok);
  Constant* ConcatLiterals(const Token* tok);
  Expr* ParseGeneric();

  void Parse();
  void ParseTranslationUnit();
  FuncDef* ParseFuncDef(Identifier* ident);


  // Expressions
  Expr* ParseExpr();
  Expr* ParsePrimaryExpr();
  QualType TryCompoundLiteral();
  Object* ParseCompoundLiteral(QualType type);
  Expr* ParsePostfixExpr();
  Expr* ParsePostfixExprTail(Expr* primExpr);
  Expr* ParseSubScripting(Expr* pointer);
  BinaryOp* ParseMemberRef(const Token* tok, int op, Expr* lhs);
  UnaryOp* ParsePostfixIncDec(const Token* tok, Expr* operand);
  FuncCall* ParseFuncCall(Expr* caller);

  Expr* ParseUnaryExpr();
  Constant* ParseSizeof();
  Constant* ParseAlignof();
  UnaryOp* ParsePrefixIncDec(const Token* tok);
  UnaryOp* ParseUnaryOp(const Token* tok, int op);

  QualType ParseTypeName();
  Expr* ParseCastExpr();
  Expr* ParseRangeExpr();
  Expr* ParseMatmulExpr();
  Expr* ParseMultiplicativeExpr();
  Expr* ParseAdditiveExpr();
  Expr* ParseShiftExpr();
  Expr* ParseRelationalExpr();
  Expr* ParseEqualityExpr();
  Expr* ParseBitiwiseAndExpr();
  Expr* ParseBitwiseXorExpr();
  Expr* ParseBitwiseOrExpr();
  Expr* ParseLogicalAndExpr();
  Expr* ParseLogicalOrExpr();
  Expr* ParseConditionalExpr();
  Expr* ParseCommaExpr();
  Expr* ParseAssignExpr();

  // Declarations
  CompoundStmt* ParseDecl();
  void ParseStaticAssert();
  QualType ParseDeclSpec(int* storageSpec, int* funcSpec, int* alignSpec);
  QualType ParseSpecQual();
  int ParseAlignas();
  Type* ParseStructUnionSpec(bool isStruct);
  StructType* ParseStructUnionDecl(StructType* type);
  void ParseBitField(StructType* structType, const Token* tok, QualType type);
  Type* ParseEnumSpec();
  Type* ParseEnumerator(ArithmType* type);
  int ParseQual();
  QualType ParsePointer(QualType typePointedTo);
  DeclInfo ParseDeclarator(QualType type);
  QualType ParseArrayFuncDeclarator(const Token* ident, QualType base);
  int ParseArrayLength();
  TileType::ShapeInt ParseTileShape();
  bool ParseParamList(FuncType::ParamList& params);
  Object* ParseParamDecl();

  QualType ParseAbstractDeclarator(QualType type);
  Identifier* ParseDirectDeclarator(QualType type,
                                    int storageSpec,
                                    int funcSpec,
                                    int align);
  // Initializer
  void ParseInitializer(Declaration* decl,
                        QualType type,
                        int offset,
                        bool designated=false,
                        bool forceBrace=false,
                        unsigned char bitFieldBegin=0,
                        unsigned char bitFieldWidth=0);
  void ParseArrayInitializer(Declaration* decl,
                             ArrayType* type,
                             int offset,
                             bool designated);
  StructType::Iterator ParseStructDesignator(StructType* type,
                                             const std::string& name);
  void ParseStructInitializer(Declaration* decl,
                              StructType* type,
                              int offset,
                              bool designated);
  bool ParseLiteralInitializer(Declaration* init,
                               ArrayType* type,
                               int offset);
  Declaration* ParseInitDeclarator(Identifier* ident);
  Declaration* ParseInitDeclaratorSub(Object* obj);

  // Statements
  Stmt* ParseStmt();
  CompoundStmt* ParseCompoundStmt(FuncType* funcType=nullptr);
  IfStmt* ParseIfStmt();
  CompoundStmt* ParseSwitchStmt();
  CompoundStmt* ParseWhileStmt();
  CompoundStmt* ParseDoStmt();
  ForStmt *ParseForStmt();
  JumpStmt* ParseGotoStmt();
  JumpStmt* ParseContinueStmt();
  JumpStmt* ParseBreakStmt();
  ReturnStmt* ParseReturnStmt();
  CompoundStmt* ParseLabelStmt(const Token* label);
  CompoundStmt* ParseCaseStmt();
  CompoundStmt* ParseDefaultStmt();
  Identifier* ProcessDeclarator(const Token* tok,
                                QualType type, const ASTNode::AttrList &attrs,
                                int storageSpec,
                                int funcSpec,
                                int align);
  // GNU extensions
  ASTNode::AttrList TryAttributeSpecList();
  void ParseAttributeSpec(ASTNode::AttrList &attrList);
  ASTNode::Attr ParseAttribute();
  bool IsTypeName(const Token* tok) const{
    if (tok->IsTypeSpecQual())
      return true;

    if (tok->IsIdentifier()) {
      auto ident = curScope_->Find(tok);
      if (ident && ident->ToTypeName())
        return true;
    }
    return false;
  }
  bool IsType(const Token* tok) const{
    if (tok->IsDecl())
      return true;

    if (tok->IsIdentifier()) {
      auto ident = curScope_->Find(tok);
      return (ident && ident->ToTypeName());
    }

    return false;
  }
  void EnsureInteger(Expr* expr) {
    if (!expr->Type()->IsInteger()) {
      Error(expr, "expect integer expression");
    }
  }

  void EnterBlock(FuncType* funcType=nullptr);
  void ExitBlock() { curScope_ = curScope_->Parent(); }
  void EnterProto() { curScope_ = new Scope(curScope_, S_PROTO); }
  void ExitProto() { curScope_ = curScope_->Parent(); }
  FuncDef* EnterFunc(Identifier* ident);
  void ExitFunc();

  LabelStmt* FindLabel(const std::string& label) {
    auto ret = curLabels_.find(label);
    if (curLabels_.end() == ret)
      return nullptr;
    return ret->second;
  }
  void AddLabel(const std::string& label, LabelStmt* labelStmt) {
    assert(nullptr == FindLabel(label));
    curLabels_[label] = labelStmt;
  }
  TranslationUnit* Unit() { return unit_; }
  FuncDef* CurFunc() { return curFunc_; }
  const TokenSequence& ts() const { return ts_; }

private:
  static bool IsBuiltin(FuncType* type);
  static bool IsBuiltin(const std::string& name);
  static Identifier* GetBuiltin(const Token* tok);
  static void DefineBuiltins();

  static FuncType* vaStartType_;
  static FuncType* vaArgType_;

  // The root of the AST
  TranslationUnit* unit_;

  TokenSequence& ts_;

  // It is not the real scope,
  // It contains all external symbols(resolved and not resolved)
  Scope* externalSymbols_;

  const Token* errTok_;
  Scope* curScope_;
  FuncDef* curFunc_;
  LabelMap curLabels_;
  LabelJumpList unresolvedJumps_;

  LabelStmt* breakDest_;
  LabelStmt* continueDest_;
  CaseLabelList* caseLabels_;
  LabelStmt* defaultLabel_;
};

#endif
