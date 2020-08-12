#pragma once

#ifndef _WGTCC_AST_H_
#define _WGTCC_AST_H_

#include "error.h"
#include "token.h"
#include "type.h"

#include <cassert>
#include <list>
#include <memory>
#include <string>


class Visitor;
template<typename T> class Evaluator;
class AddrEvaluator;
class Generator;

class Scope;
class Parser;
class ASTNode;
class Token;
class TokenSequence;

// Expressions
class Expr;
class BinaryOp;
class UnaryOp;
class ConditionalOp;
class FuncCall;
class TempVar;
class Constant;

class Identifier;
class Object;
struct Initializer;
class Declaration;
class Enumerator;

// Statements
class Stmt;
class IfStmt;
class ForStmt;
class JumpStmt;
class LabelStmt;
class EmptyStmt;
class CompoundStmt;
class FuncDef;
class TranslationUnit;


/*
 * AST Node
 */

class ASTNode {
public:
  struct Attr{

    enum KindT{
      MULTIPLEOF,
      ALIGNED,
      NOALIAS,
      READONLY,
      WRITEONLY,
      RETUNE,
    };

    KindT kind;
    std::vector<Expr*> vals;
  };
  using AttrList = std::vector<Attr>;

public:
  virtual ~ASTNode() {}
  virtual void Accept(Visitor* v) = 0;

protected:
  ASTNode() {}

  MemPool* pool_ {nullptr};
};

using ExtDecl = ASTNode;


/*
 * Statements
 */

class Stmt : public ASTNode {
public:
  virtual ~Stmt() {}

protected:
   Stmt() {}
};


class EmptyStmt : public Stmt {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;

public:
  static EmptyStmt* New();
  virtual ~EmptyStmt() {}
  virtual void Accept(Visitor* v);

protected:
  EmptyStmt() {}
};


class LabelStmt : public Stmt {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;

public:
  static LabelStmt* New();
  ~LabelStmt() {}
  virtual void Accept(Visitor* v);
  std::string Repr() const { return ".L" + std::to_string(tag_); }

protected:
  LabelStmt(): tag_(GenTag()) {}

private:
  static int GenTag() {
    static int tag = 0;
    return ++tag;
  }

  int tag_; // 使用整型的tag值，而不直接用字符串
};


class IfStmt : public Stmt {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;
public:
  static IfStmt* New(Expr* cond, Stmt* then, Stmt* els=nullptr);
  virtual ~IfStmt() {}
  virtual void Accept(Visitor* v);

protected:
  IfStmt(Expr* cond, Stmt* then, Stmt* els = nullptr)
      : cond_(cond), then_(then), else_(els) {}

private:
  Expr* cond_;
  Stmt* then_;
  Stmt* else_;
};

class ForStmt: public Stmt {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;
public:
  static ForStmt* New(Stmt* body, Stmt* init = nullptr, Expr* cond = nullptr, Expr* step = nullptr);
  virtual ~ForStmt() {}
  virtual void Accept(Visitor* v);

protected:
  ForStmt(Stmt* body, Stmt* init = nullptr, Expr* cond = nullptr, Expr* step = nullptr)
      : body_(body), init_(init), cond_(cond), step_(step) {}

private:
  Stmt* body_;
  Stmt* init_;
  Expr* cond_;
  Expr* step_;
};

class JumpStmt : public Stmt {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;

public:
  static JumpStmt* New(LabelStmt* label);
  virtual ~JumpStmt() {}
  virtual void Accept(Visitor* v);
  void SetLabel(LabelStmt* label) { label_ = label; }

protected:
  JumpStmt(LabelStmt* label): label_(label) {}

private:
  LabelStmt* label_;
};


class ReturnStmt: public Stmt {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;

public:
  static ReturnStmt* New(Expr* expr);
  virtual ~ReturnStmt() {}
  virtual void Accept(Visitor* v);

protected:
  ReturnStmt(::Expr* expr): expr_(expr) {}

private:
  ::Expr* expr_;
};


using StmtList = std::list<Stmt*>;

class CompoundStmt : public Stmt {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;

public:
  static CompoundStmt* New(StmtList& stmts, ::Scope* scope=nullptr);
  virtual ~CompoundStmt() {}
  virtual void Accept(Visitor* v);
  StmtList& Stmts() { return stmts_; }
  ::Scope* Scope() { return scope_; }

protected:
  CompoundStmt(const StmtList& stmts, ::Scope* scope=nullptr)
      : stmts_(stmts), scope_(scope) {}

private:
  StmtList stmts_;
  ::Scope* scope_;
};


struct Initializer {
  Initializer(Type* type,
              int offset,
              Expr* expr,
              unsigned char bitFieldBegin=0,
              unsigned char bitFieldWidth=0)
      : type_(type),
        offset_(offset),
        bitFieldBegin_(bitFieldBegin),
        bitFieldWidth_(bitFieldWidth),
        expr_(expr) {}

  bool operator<(const Initializer& rhs) const;

  // It could be the object it self or, it will be the member
  // that was initialized
  Type* type_;
  int offset_;
  unsigned char bitFieldBegin_;
  unsigned char bitFieldWidth_;

  Expr* expr_;
};


using InitList = std::set<Initializer>;

class Declaration: public Stmt {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;

public:
  static Declaration* New(Object* obj);
  virtual ~Declaration() {}
  virtual void Accept(Visitor* v);
  InitList& Inits() { return inits_; }
  Object* Obj() { return obj_; }
  void AddInit(Initializer init);

protected:
  Declaration(Object* obj): obj_(obj) {}

  Object* obj_;
  InitList inits_;
};


/*
 * Expr
 *  BinaryOp
 *  UnaryOp
 *  ConditionalOp
 *  FuncCall
 *  Constant
 *  Identifier
 *  Object
 *  TempVar
 */

class Expr : public Stmt {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;
  friend class LValAssigner;

public:
  virtual ~Expr() {}
  ::Type* Type() { return type_.GetPtr(); }
  virtual bool IsLVal() = 0;
  virtual void TypeChecking() = 0;
  void EnsureCompatible(const QualType lhs, const QualType rhs) const;
  void EnsureCompatibleOrVoidPointer(const QualType lhs,
                                     const QualType rhs) const;
  const Token* Tok() const { return tok_; }
  void SetTok(const Token* tok) { tok_ = tok; }

  static Expr* MayCast(Expr* expr);
  static Expr* MayCast(Expr* expr, QualType desType);
  static ::Type* TryExtractScalarType(Expr* loc, Expr *operand);
  static ::Type* ScalarOrLikeTile(Expr* operand, ::Type* ty);

  virtual bool IsNullPointerConstant() const { return false; }
  bool IsConstQualified() const { return type_.IsConstQualified(); }
  bool IsRestrictQualified() const { return type_.IsRestrictQualified(); }
  bool IsVolatileQualified() const { return type_.IsVolatileQualified(); }

protected:
  // You can construct a expression without specifying a type,
  // then the type should be evaluated in TypeChecking()
  Expr(const Token* tok, QualType type): tok_(tok), type_(type) {}

  const Token* tok_;
  QualType type_;
};


/*
 * '+', '-', '*', '/', '%', '<', '>', '<<', '>>', '|', '&', '^'
 * '=',(复合赋值运算符被拆分为两个运算)
 * '==', '!=', '<=', '>=',
 * '&&', '||'
 * '['(下标运算符), '.'(成员运算符)
 * ','(逗号运算符),
 */
class BinaryOp : public Expr {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;
  friend class LValAssigner;
  friend class Declaration;

public:
  static BinaryOp* New(const Token* tok, Expr* lhs, Expr* rhs);
  static BinaryOp* New(const Token* tok, int op, Expr* lhs, Expr* rhs);
  virtual ~BinaryOp() {}
  virtual void Accept(Visitor* v);

  // Member ref operator is a lvalue
  virtual bool IsLVal() {
    switch (op_) {
    case '.': return !Type()->ToArray() && lhs_->IsLVal();
    case ']': return !Type()->ToArray();
    case Token::MASKED_DEREF: return true;
    default: return false;
    }
  }
  ArithmType* Convert();
  static void Broadcast(Expr* loc, Expr*& lhs, Expr*& rhs, QualType &type);

  virtual void TypeChecking();
  void SubScriptingOpTypeChecking();
  void MemberRefOpTypeChecking();
  void MultiOpTypeChecking();
  void AdditiveOpTypeChecking();
  void ShiftOpTypeChecking();
  void RangeOpTypeChecking();
  void MatmulOpTypeChecking();
  void MaskedDerefOpTypeChecking();
  void RelationalOpTypeChecking();
  void EqualityOpTypeChecking();
  void BitwiseOpTypeChecking();
  void LogicalOpTypeChecking();
  void AssignOpTypeChecking();
  void CommaOpTypeChecking();

protected:
  BinaryOp(const Token* tok, int op, Expr* lhs, Expr* rhs)
      : Expr(tok, nullptr), op_(op) {
        lhs_ = lhs, rhs_ = rhs;
        if (op != '.') {
          lhs_ = MayCast(lhs);
          rhs_ = MayCast(rhs);
        }
      }

  int op_;
  Expr* lhs_;
  Expr* rhs_;
};


/*
 * Unary Operator:
 * '++' (prefix/postfix)
 * '--' (prefix/postfix)
 * '&'  (ADDR)
 * '*'  (DEREF)
 * '+'  (PLUS)
 * '-'  (MINUS)
 * '~'
 * '!'
 * CAST // like (int)3
 */
class UnaryOp : public Expr {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;
  friend class LValAssigner;

public:
  static UnaryOp* New(int op, Expr* operand, QualType type=nullptr, int info=0);
  virtual ~UnaryOp() {}
  virtual void Accept(Visitor* v);
  virtual bool IsLVal();
  ::Type *Convert();
  static int encodeRed(int ax, int tag);
  static void decodeRed(int info, int& ax, int& tag);
  void TypeChecking();
  void IncDecOpTypeChecking();
  void AddrOpTypeChecking();
  void DerefOpTypeChecking();
  void ReduceOpTypeChecking();
  void UnaryArithmOpTypeChecking();
  void BitcastOpTypeChecking();
  void CastOpTypeChecking();
  void IntrinsicOpTypeChecking();

protected:
  UnaryOp(int op, Expr* operand, QualType type=nullptr, int info=0)
    : Expr(operand->Tok(), type), op_(op), info_(info) {
      operand_ = operand;
      if (op_ != Token::CAST && op_ != Token::ADDR) {
        operand_ = MayCast(operand);
      }
    }

  int op_;
  int info_;
  Expr* operand_;
};

class TransOp: public Expr {
  friend class Generator;

public:
  using PermInt = std::vector<int>;

public:
  static TransOp* New(const PermInt& perm, Expr* operand);
  const PermInt& getPerm() const { return perm_; }
  void Accept(Visitor* v);
  bool IsLVal() { return false; }
  void TypeChecking();

protected:
  TransOp(const PermInt& perm, Expr* operand)
    : Expr(operand->Tok(), nullptr), operand_(operand), perm_(perm) {}

private:
  Expr* operand_;
  PermInt perm_;
};


// cond ? true ： false
class ConditionalOp : public Expr {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;

public:
  static ConditionalOp* New(const Token* tok,
      Expr* cond, Expr* exprTrue, Expr* exprFalse);
  virtual ~ConditionalOp() {}
  virtual void Accept(Visitor* v);
  virtual bool IsLVal() { return false; }
  ArithmType* Convert();
  virtual void TypeChecking();

protected:
  ConditionalOp(Expr* cond, Expr* exprTrue, Expr* exprFalse)
      : Expr(cond->Tok(), nullptr), cond_(MayCast(cond)),
        exprTrue_(MayCast(exprTrue)), exprFalse_(MayCast(exprFalse)) {}

private:
  Expr* cond_;
  Expr* exprTrue_;
  Expr* exprFalse_;
};


class FuncCall : public Expr {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;

public:
  using ArgList = std::vector<Expr*>;

public:
  static FuncCall* New(Expr* designator, const ArgList& args);
  ~FuncCall() {}
  virtual void Accept(Visitor* v);

  // A function call is ofcourse not lvalue
  virtual bool IsLVal() { return false; }
  ArgList* Args() { return &args_; }
  Expr* Designator() { return designator_; }
  const std::string& Name() const { return tok_->str_; }
  ::FuncType* FuncType() { return designator_->Type()->ToFunc(); }
  virtual void TypeChecking();

protected:
  FuncCall(Expr* designator, const ArgList& args)
    : Expr(designator->Tok(), nullptr),
      designator_(designator), args_(args) {}

  Expr* designator_;
  ArgList args_;
};


class Constant: public Expr {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;

public:
  static Constant* New(const Token* tok, int tag, long val);
  static Constant* New(const Token* tok, int tag, double val);
  static Constant* New(const Token* tok, int tag, const std::string* val);
  ~Constant() {}
  virtual void Accept(Visitor* v);
  virtual bool IsLVal() { return false; }
  virtual void TypeChecking() {}

  long IVal() const { return ival_; }
  double FVal() const { return fval_; }
  const std::string* SVal() const { return sval_; }
  std::string SValRepr() const;
  std::string Repr() const { return std::string(".LC") + std::to_string(id_); }

protected:
  Constant(const Token* tok, QualType type, long val)
      : Expr(tok, type), ival_(val) {}
  Constant(const Token* tok, QualType type, double val)
      : Expr(tok, type), fval_(val) {}
  Constant(const Token* tok, QualType type, const std::string* val)
      : Expr(tok, type), sval_(val) {}

  union {
    long ival_;
    double fval_;
    struct {
      long id_;
      const std::string* sval_;
    };
  };
};


class TempVar : public Expr {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;

public:
  static TempVar* New(QualType type);
  virtual ~TempVar() {}
  virtual void Accept(Visitor* v);
  virtual bool IsLVal() { return true; }
  virtual void TypeChecking() {}

protected:
  TempVar(QualType type): Expr(nullptr, type), tag_(GenTag()) {}

private:
  static int GenTag() {
    static int tag = 0;
    return ++tag;
  }

  int tag_;
};


enum Linkage {
  L_NONE,
  L_EXTERNAL,
  L_INTERNAL,
};


class Identifier: public Expr {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;
  friend class LValAssigner;

public:
  static Identifier* New(const Token* tok, QualType type, Linkage linkage, const AttrList& attrList={});
  virtual ~Identifier() {}
  virtual void Accept(Visitor* v);
  virtual bool IsLVal() { return false; }
  virtual Object* ToObject() { return nullptr; }
  virtual Enumerator* ToEnumerator() { return nullptr; }

   // An identifer can be:
   //   object, sturct/union/enum tag, typedef name, function, label.
   Identifier* ToTypeName() {
    // A typename has no linkage
    // And a function has external or internal linkage
    if (ToObject() || ToEnumerator() || linkage_ != L_NONE)
      return nullptr;
    return this;
  }
  virtual const std::string Name() const { return tok_->str_; }
  enum Linkage Linkage() const { return linkage_; }
  void SetLinkage(enum Linkage linkage) { linkage_ = linkage; }
  virtual void TypeChecking() {}

protected:
  Identifier(const Token* tok, QualType type, enum Linkage linkage, const AttrList& attrList={})
      : Expr(tok, type), linkage_(linkage), attrList_(attrList) {}

  // An identifier has property linkage
  enum Linkage linkage_;
  AttrList attrList_;
};


class Enumerator: public Identifier {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;

public:
  static Enumerator* New(const Token* tok, int val);
  virtual ~Enumerator() {}
  virtual void Accept(Visitor* v);
  virtual Enumerator* ToEnumerator() { return this; }
  int Val() const { return cons_->IVal(); }

protected:
  Enumerator(const Token* tok, int val)
      : Identifier(tok, ArithmType::New(T_INT), L_NONE),
        cons_(Constant::New(tok, T_INT, (long)val)) {}

  Constant* cons_;
};


class Object : public Identifier {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;
  friend class LValAssigner;

public:
  static Object* New(const Token* tok,
                     QualType type,
                     int storage=0,
                     enum Linkage linkage=L_NONE,
                     unsigned char bitFieldBegin=0,
                     unsigned char bitFieldWidth=0,
                     const AttrList& attrList={});
  static Object* NewAnony(const Token* tok,
                          QualType type,
                          int storage=0,
                          enum Linkage linkage=L_NONE,
                          unsigned char bitFieldBegin=0,
                          unsigned char bitFieldWidth=0,
                          const AttrList& attrList={});
  ~Object() {}
  virtual void Accept(Visitor* v);
  virtual Object* ToObject() { return this; }
  virtual bool IsLVal() {
    // TODO(wgtdkp): not all object is lval?
    return true;
  }
  bool IsStatic() const {
    return (Storage() & S_STATIC) || (Linkage() != L_NONE);
  }
  int Storage() const { return storage_; }
  void SetStorage(int storage) { storage_ = storage; }
  int Align() const { return align_; }
  void SetAlign(int align) {
    assert(align > 0);
    // Allowing reduce alignment to implement __attribute__((packed))
    //if (align < align_)
    //  Error(this, "alignment specifier cannot reduce alignment");
    align_ = align;
  }
  int Offset() const { return offset_; }
  void SetOffset(int offset) { offset_ = offset; }
  Declaration* Decl() { return decl_; }
  void SetDecl(Declaration* decl) { decl_ = decl; }
  const AttrList& GetAttrList() const { return attrList_; }
  unsigned char BitFieldBegin() const { return bitFieldBegin_; }
  unsigned char BitFieldEnd() const { return bitFieldBegin_ + bitFieldWidth_; }
  unsigned char BitFieldWidth() const { return bitFieldWidth_; }
  static unsigned long BitFieldMask(Object* bitField) {
    return BitFieldMask(bitField->bitFieldBegin_, bitField->bitFieldWidth_);
  }
  static unsigned long BitFieldMask(unsigned char begin, unsigned char width) {
    auto end = begin + width;
    return ((0xFFFFFFFFFFFFFFFFUL << (64 - end)) >> (64 - width)) << begin;
  }

  bool HasInit() const { return decl_ && decl_->Inits().size(); }
  bool Anonymous() const { return anonymous_; }
  virtual const std::string Name() const { return Identifier::Name(); }
  std::string Repr() const {
    assert(IsStatic() || anonymous_);
    if (anonymous_)
      return "anonymous." + std::to_string(id_);
    if (linkage_ == L_NONE)
      return Name() + "." + std::to_string(id_);
    return Name();
  }

protected:
  Object(const Token* tok,
         QualType type,
         int storage=0,
         enum Linkage linkage=L_NONE,
         unsigned char bitFieldBegin=0,
         unsigned char bitFieldWidth=0,
         const AttrList& attrList={})
      : Identifier(tok, type, linkage),
        storage_(storage),
        offset_(0),
        align_(type->Align()),
        decl_(nullptr),
        bitFieldBegin_(bitFieldBegin),
        bitFieldWidth_(bitFieldWidth),
        anonymous_(false),
        attrList_(attrList){}

private:
  int storage_;
  int offset_;
  int align_;

  Declaration* decl_;

  unsigned char bitFieldBegin_;
  // 0 means it's not a bitfield
  unsigned char bitFieldWidth_;

  bool anonymous_;
  long id_ {0};

  AttrList attrList_;
};


/*
 * Declaration
 */

class FuncDef : public ExtDecl {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;

public:
  using ParamList = std::vector<Object*>;

public:
  static FuncDef* New(Identifier* ident, LabelStmt* retLabel);
  virtual ~FuncDef() {}
  virtual void Accept(Visitor* v);
  ::FuncType* FuncType() { return ident_->Type()->ToFunc(); }
  CompoundStmt* Body() { return body_; }
  void SetBody(CompoundStmt* body) { body_ = body; }
  std::string Name() const { return ident_->Name(); }
  enum Linkage Linkage() { return ident_->Linkage(); }

protected:
  FuncDef(Identifier* ident, LabelStmt* retLabel)
      : ident_(ident), retLabel_(retLabel) {}

private:
  Identifier* ident_;
  LabelStmt* retLabel_;
  CompoundStmt* body_;
};


using ExtDeclList = std::list<ExtDecl*>;

class TranslationUnit : public ASTNode {
  template<typename T> friend class Evaluator;
  friend class AddrEvaluator;
  friend class Generator;

public:
  static TranslationUnit* New() { return new TranslationUnit();}
  virtual ~TranslationUnit() {}
  virtual void Accept(Visitor* v);
  void Add(ExtDecl* extDecl) { extDecls_.push_back(extDecl); }
  ExtDeclList& ExtDecls() { return extDecls_; }
  const ExtDeclList& ExtDecls() const { return extDecls_; }

private:
  TranslationUnit() {}

  ExtDeclList extDecls_;
};

#endif
