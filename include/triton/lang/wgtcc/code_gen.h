#ifndef _WGTCC_CODE_GEN_H_
#define _WGTCC_CODE_GEN_H_

#include "ast.h"
#include "visitor.h"


class Parser;
struct Addr;
struct ROData;
template<> class Evaluator<Addr>;
struct StaticInitializer;

using TypeList = std::vector<Type*>;
using LocationList = std::vector<std::string>;
using RODataList = std::vector<ROData>;
using StaticInitList = std::vector<StaticInitializer>;


enum class ParamClass {
  INTEGER,
  SSE,
  SSEUP,
  X87,
  X87_UP,
  COMPLEX_X87,
  NO_CLASS,
  MEMORY
};

struct ParamLocations {
  LocationList locs_;
  size_t regCnt_;
  size_t xregCnt_;
};

struct ROData {
  ROData(long ival, int align): ival_(ival), align_(align) {
    label_ = ".LC" + std::to_string(GenTag());
  }

  explicit ROData(const std::string& sval): sval_(sval), align_(1) {
    label_ = ".LC" + std::to_string(GenTag());
  }

  ~ROData() {}

  std::string sval_;
  long ival_;
  int align_;
  std::string label_;

private:
  static long GenTag() {
    static long tag = 0;
    return tag++;
  }
};


struct ObjectAddr {
  explicit ObjectAddr(int offset)
      : ObjectAddr("", "%rbp", offset) {}

  ObjectAddr(const std::string& label, const std::string& base, int offset)
      : label_(label), base_(base), offset_(offset) {}

  std::string Repr() const;

  std::string label_;
  std::string base_;
  int offset_;
  unsigned char bitFieldBegin_ {0};
  unsigned char bitFieldWidth_ {0};
};


struct StaticInitializer {
  int offset_;
  int width_;
  long val_;
  std::string label_;
};


class Generator: public Visitor {
  friend class Evaluator<Addr>;
public:
  Generator() {}

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


  static void SetInOut(Parser* parser, FILE* outFile) {
    parser_ = parser;
    outFile_ = outFile;
  }

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

  void CopyStruct(ObjectAddr desAddr, int width);

  std::string ConsLabel(Constant* cons);

  ParamLocations GetParamLocations(const TypeList& types, bool retStruct);
  void GetParamRegOffsets(int& gpOffset, int& fpOffset,
      int& overflow, FuncType* funcType);

  void Emit(const std::string& str) {
    fprintf(outFile_, "\t%s\n", str.c_str());
  }

  void Emit(const std::string& inst,
            const std::string& src,
            const std::string& des) {
    Emit(inst + "\t" + src + ", " + des);
  }

  void Emit(const std::string& inst,
            int imm,
            const std::string& reg) {
    Emit(inst + "\t$" + std::to_string(imm) + ", " + reg);
  }

  void Emit(const std::string& inst,
            const std::string& des) {
    Emit(inst + "\t" + des);
  }

  void Emit(const std::string& inst,
            const LabelStmt* label) {
    Emit(inst + "\t" + label->Repr());
  }

  void Emit(const std::string& inst,
            const ObjectAddr& src,
            const ObjectAddr& des) {
    Emit(inst, src.Repr(), des.Repr());
  }

  void Emit(const std::string& inst,
            const std::string& src,
            const ObjectAddr& des) {
    Emit(inst, src, des.Repr());
  }

  void Emit(const std::string& inst,
            const ObjectAddr& src,
            const std::string& des) {
    Emit(inst, src.Repr(), des);
  }

  void EmitLabel(const std::string& label);
  void EmitZero(ObjectAddr addr, int width);
  void EmitLoad(const std::string& addr, Type* type);
  void EmitLoad(const std::string& addr, int width, bool flt);
  void EmitStore(const ObjectAddr& addr, Type* type);
  void EmitStore(const std::string& addr, Type* type);
  void EmitStore(const std::string& addr, int width, bool flt);
  void EmitLoadBitField(const std::string& addr, Object* bitField);
  void EmitStoreBitField(const ObjectAddr& addr, Type* type);
  void EmitLoc(Expr* expr);

  int Push(Type* type);
  int Push(const std::string& reg);
  int Pop(const std::string& reg);

  void Spill(bool flt);

  void Restore(bool flt);

  void Save(bool flt);

  void Exchange(bool flt);

protected:
  static const std::string* last_file;
  static Parser* parser_;
  static FILE* outFile_;
  static RODataList rodatas_;
  static int offset_;

  // The address that store the register %rdi,
  // when the return value is a struct/union
  static int retAddrOffset_;
  static FuncDef* curFunc_;

  static std::vector<Declaration*> staticDecls_;
};


class LValGenerator: public Generator {
public:
  LValGenerator() {}

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

  ObjectAddr GenExpr(Expr* expr) {
    expr->Accept(this);
    return addr_;
  }

private:
  ObjectAddr addr_ {"", "", 0};
};

#endif
