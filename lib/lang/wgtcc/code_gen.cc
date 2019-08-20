#include "triton/lang/wgtcc/code_gen.h"

#include "triton/lang/wgtcc/evaluator.h"
#include "triton/lang/wgtcc/parser.h"
#include "triton/lang/wgtcc/token.h"

#include <cstdarg>
#include <queue>
#include <set>


extern std::string filename_in;
extern std::string filename_out;
extern bool debug;

const std::string* Generator::last_file = nullptr;
Parser* Generator::parser_ = nullptr;
FILE* Generator::outFile_ = nullptr;
RODataList Generator::rodatas_;
std::vector<Declaration*> Generator::staticDecls_;
int Generator::offset_ = 0;
int Generator::retAddrOffset_ = 0;
FuncDef* Generator::curFunc_ = nullptr;


/*
 * Register usage:
 *  xmm0: accumulator of floating datas;
 *  xmm8: temp register for param passing(xmm0)
 *  xmm9: source operand register;
 *  xmm10: tmp register for floating data swap;
 *  rax: accumulator;
 *  r12, r13: temp register for rdx and rcx
 *  r11: source operand register;
 *  r10: base register when LValGenerator eval the address.
 *  rcx: tempvar register, like the tempvar of 'switch'
 *       temp register for struct copy
 */

static std::vector<const char*> regs {
  "%rdi", "%rsi", "%rdx",
  "%rcx", "%r8", "%r9"
};

static std::vector<const char*> xregs {
  "%xmm0", "%xmm1", "%xmm2", "%xmm3",
  "%xmm4", "%xmm5", "%xmm6", "%xmm7"
};


static ParamClass Classify(Type* paramType, int offset=0) {
  if (paramType->IsInteger() || paramType->ToPointer()
      || paramType->ToArray()) {
    return ParamClass::INTEGER;
  }

  if (paramType->ToArithm()) {
    auto type = paramType->ToArithm();
    if (type->Tag() == T_FLOAT || type->Tag() == T_DOUBLE)
      return ParamClass::SSE;
    if (type->Tag() == (T_LONG | T_DOUBLE)) {
      // TODO(wgtdkp):
      return ParamClass::SSE;
      assert(false);
      return ParamClass::X87;
    }

    // TODO(wgtdkp):
    assert(false);
    // It is complex
    if ((type->Tag() & T_LONG) && (type->Tag() & T_DOUBLE))
      return ParamClass::COMPLEX_X87;
  }
  auto type = paramType->ToStruct();
  assert(type);
  return ParamClass::MEMORY;
  // TODO(wgtdkp): Support agrregate type
  assert(false);
  /*
  auto type = paramType->ToStruct();
  assert(type);

  if (type->Width() > 4 * 8)
    return PC_MEMORY;

  std::vector<ParamClass> classes;
  int cnt = (type->Width() + 7) / 8;
  for (int i = 0; i < cnt; ++i) {
    auto  types = FieldsIn8Bytes(type, i);
    assert(types.size() > 0);

    auto fieldClass = (types.size() == 1)
        ? PC_NO_CLASS: FieldClass(types, 0);
    classes.push_back(fieldClass);

  }

  bool sawX87 = false;
  for (int i = 0; i < classes.size(); ++i) {
    if (classes[i] == PC_MEMORY)
      return PC_MEMORY;
    if (classes[i] == PC_X87_UP && sawX87)
      return PC_MEMORY;
    if (classes[i] == PC_X87)
      sawX87 = true;
  }
  */
  return ParamClass::NO_CLASS; // Make compiler happy
}


std::string Generator::ConsLabel(Constant* cons) {
  if (cons->Type()->IsInteger()) {
    return "$" + std::to_string(cons->IVal());
  } else if (cons->Type()->IsFloat()) {
    double valsd = cons->FVal();
    float  valss = valsd;
    // TODO(wgtdkp): Add rodata
    auto width = cons->Type()->Width();
    long val = (width == 4)? *reinterpret_cast<int*>(&valss):
                             *reinterpret_cast<long*>(&valsd);
    const ROData& rodata = ROData(val, width);
    rodatas_.push_back(rodata);
    return rodata.label_;
  } else { // Literal
    const ROData& rodata = ROData(cons->SValRepr());
    rodatas_.push_back(rodata);
    return rodata.label_; // Return address
  }
}


static const char* GetLoad(int width, bool flt=false) {
  switch (width) {
  case 1: return "movzbq";
  case 2: return "movzwq";
  case 4: return !flt ? "movl": "movss";
  case 8: return !flt ? "movq": "movsd";
  default: assert(false); return nullptr;
  }
}


static std::string GetInst(const std::string& inst, int width, bool flt) {
  if (flt)  {
    return inst + (width == 4 ? "ss": "sd");
  } else {
    switch (width) {
    case 1: return inst + "b";
    case 2: return inst + "w";
    case 4: return inst + "l";
    case 8: return inst + "q";
    default: assert(false);
    }
    return inst; // Make compiler happy
  }
}


static std::string GetInst(const std::string& inst, Type* type) {
  assert(type->IsScalar());
  return GetInst(inst, type->Width(), type->IsFloat());
}


static std::string GetReg(int width) {
  switch (width) {
  case 1: return "%al";
  case 2: return "%ax";
  case 4: return "%eax";
  case 8: return "%rax";
  default: assert(false); return "";
  }
}


static std::string GetDes(int width, bool flt) {
  if (flt) {
    return "%xmm0";
  }
  return GetReg(width);
}


static std::string GetSrc(int width, bool flt) {
  if (flt) {
    return "%xmm9";
  }
  switch (width) {
  case 1: return "%r11b";
  case 2: return "%r11w";
  case 4: return "%r11d";
  case 8: return "%r11";
  default: assert(false); return "";
  }
}


// The 'reg' always be 8 bytes
int Generator::Push(const std::string& reg) {
  offset_ -= 8;
  auto mov = reg[1] == 'x' ? "movsd": "movq";
  Emit(mov, reg, ObjectAddr(offset_));
  return offset_;
}


int Generator::Push(Type* type) {
  if (type->IsFloat()) {
    return Push("%xmm0");
  } else if (type->IsScalar()) {
    return Push("%rax");
  } else {
    offset_ -= type->Width();
    offset_ = Type::MakeAlign(offset_, 8);
    CopyStruct({"", "%rbp", offset_}, type->Width());
    return offset_;
  }
}


// The 'reg' must be 8 bytes
int Generator::Pop(const std::string& reg) {
  auto mov = reg[1] == 'x' ? "movsd": "movq";
  Emit(mov, ObjectAddr(offset_), reg);
  offset_ += 8;
  return offset_;
}


void Generator::Spill(bool flt) {
  Push(flt ? "%xmm0": "%rax");
}


void Generator::Restore(bool flt) {
  const auto& src = GetSrc(8, flt);
  const auto& des = GetDes(8, flt);
  const auto& inst = GetInst("mov", 8, flt);
  Emit(inst, des, src);
  Pop(des);
}


void Generator::Save(bool flt) {
  if (flt) {
    Emit("movsd", "%xmm0", "%xmm9");
  } else {
    Emit("movq", "%rax", "%r11");
  }
}


/*
 * Operator/Instruction mapping:
 * +  add
 * -  sub
 * *  mul
 * /  div
 * %  div
 * << sal
 * >> sar
 * |  or
 * &  and
 * ^  xor
 * =  mov
 * <  cmp, setl, movzbq
 * >  cmp, setg, movzbq
 * <= cmp, setle, movzbq
 * >= cmp, setle, movzbq
 * == cmp, sete, movzbq
 * != cmp, setne, movzbq
 * && GenAndOp
 * || GenOrOp
 * ]  GenSubScriptingOp
 * .  GenMemberRefOp
 */
void Generator::VisitBinaryOp(BinaryOp* binary) {
  EmitLoc(binary);
  auto op = binary->op_;

  if (op == '=')
    return GenAssignOp(binary);
  if (op == Token::LOGICAL_AND)
    return GenAndOp(binary);
  if (op == Token::LOGICAL_OR)
    return GenOrOp(binary);
  if (op == '.')
    return GenMemberRefOp(binary);
  if (op == ',')
    return GenCommaOp(binary);
  // Why lhs_->Type() ?
  // Because, the type of pointer subtraction is arithmetic type
  if (binary->lhs_->Type()->ToPointer() &&
      (op == '+' || op == '-')) {
    return GenPointerArithm(binary);
  }

  // Careful: for compare operator, the type of the expression
  // is always integer, while the type of lhs and rhs could be float
  // After convertion, lhs and rhs always has the same type
  auto type = binary->lhs_->Type();
  auto width = type->Width();
  auto flt = type->IsFloat();
  auto sign = !type->IsUnsigned();

  Visit(binary->lhs_);
  Spill(flt);
  Visit(binary->rhs_);
  Restore(flt);

  const char* inst = nullptr;

  switch (op) {
  case '*': return GenMulOp(width, flt, sign);
  case '/': case '%': return GenDivOp(flt, sign, width, op);
  case '<':
    return GenCompOp(width, flt, (flt || !sign) ? "setb": "setl");
  case '>':
    return GenCompOp(width, flt, (flt || !sign) ? "seta": "setg");
  case Token::LE:
    return GenCompOp(width, flt, (flt || !sign) ? "setbe": "setle");
  case Token::GE:
    return GenCompOp(width, flt, (flt || !sign) ? "setae": "setge");
  case Token::EQ:
    return GenCompOp(width, flt, "sete");
  case Token::NE:
    return GenCompOp(width, flt, "setne");

  case '+': inst = "add"; break;
  case '-': inst = "sub"; break;
  case '|': inst = "or"; break;
  case '&': inst = "and"; break;
  case '^': inst = "xor"; break;
  case Token::LEFT: case Token::RIGHT:
    inst = op == Token::LEFT ? "sal": (sign ? "sar": "shr");
    Emit("movq %r11, %rcx");
    Emit(GetInst(inst, width, flt), "%cl", GetDes(width, flt));
    return;
  }
  Emit(GetInst(inst, width, flt), GetSrc(width, flt), GetDes(width, flt));
}


void Generator::GenCommaOp(BinaryOp* comma) {
  VisitExpr(comma->lhs_);
  VisitExpr(comma->rhs_);
}


void Generator::GenMulOp(int width, bool flt, bool sign) {
  auto inst = flt ? "mul": (sign ? "imul": "mul");

  if (flt) {
    Emit(GetInst(inst, width, flt), "%xmm9", "%xmm0");
  } else {
    Emit(GetInst(inst, width, flt), GetSrc(width, flt));
  }
}


void Generator::GenCompZero(Type* type) {
  auto width = type->Width();
  auto flt = type->IsFloat();

  if (!flt) {
    Emit("cmp", "$0", GetReg(width));
  } else {
    Emit("pxor", "%xmm9", "%xmm9");
    auto cmp = width == 8 ? "ucomisd": "ucomiss";
    Emit(cmp, "%xmm9", "%xmm0");
  }
}


void Generator::GenAndOp(BinaryOp* andOp) {
  VisitExpr(andOp->lhs_);
  GenCompZero(andOp->lhs_->Type());

  auto labelFalse = LabelStmt::New();
  Emit("je", labelFalse);

  VisitExpr(andOp->rhs_);
  GenCompZero(andOp->rhs_->Type());

  Emit("je", labelFalse);

  Emit("movq", "$1", "%rax");
  auto labelTrue = LabelStmt::New();
  Emit("jmp", labelTrue);
  EmitLabel(labelFalse->Repr());
  Emit("xorq", "%rax", "%rax"); // Set %rax to 0
  EmitLabel(labelTrue->Repr());
}


void Generator::GenOrOp(BinaryOp* orOp) {
  VisitExpr(orOp->lhs_);
  GenCompZero(orOp->lhs_->Type());

  auto labelTrue = LabelStmt::New();
  Emit("jne", labelTrue);

  VisitExpr(orOp->rhs_);
  GenCompZero(orOp->rhs_->Type());

  Emit("jne", labelTrue);

  Emit("xorq", "%rax", "%rax"); // Set %rax to 0
  auto labelFalse = LabelStmt::New();
  Emit("jmp", labelFalse);
  EmitLabel(labelTrue->Repr());
  Emit("movq", "$1", "%rax");
  EmitLabel(labelFalse->Repr());
}


void Generator::GenMemberRefOp(BinaryOp* ref) {
  // As the lhs will always be struct/union
  auto addr = LValGenerator().GenExpr(ref->lhs_);
  const auto& name = ref->rhs_->Tok()->str_;
  auto structType = ref->lhs_->Type()->ToStruct();
  auto member = structType->GetMember(name);

  addr.offset_ += member->Offset();

  if (!ref->Type()->IsScalar()) {
    Emit("leaq", addr, "%rax");
  } else {
    if (member->BitFieldWidth()) {
      EmitLoadBitField(addr.Repr(), member);
    } else {
      EmitLoad(addr.Repr(), ref->Type());
    }
  }
}


void Generator::EmitLoadBitField(const std::string& addr, Object* bitField) {
  auto type = bitField->Type()->ToArithm();
  assert(type && type->IsInteger());

  EmitLoad(addr, type);
  Emit("andq", Object::BitFieldMask(bitField), "%rax");

  auto shiftRight = (type->Tag() & T_UNSIGNED) ? "shrq": "sarq";
  auto left = 64 - bitField->bitFieldBegin_ - bitField->bitFieldWidth_;
  auto right = 64 - bitField->bitFieldWidth_;
  Emit("salq", left, "%rax");
  Emit(shiftRight, right, "%rax");
}


// FIXME(wgtdkp): for combined assignment operator, if the rvalue expr
// has some side-effect, the rvalue will be evaluated twice!
void Generator::GenAssignOp(BinaryOp* assign) {
  // The base register of addr is %r10, %rip, %rbp
  auto addr = LValGenerator().GenExpr(assign->lhs_);
  // Base register of static object maybe %rip
  // Visit rhs_ may changes r10
  if (addr.base_ == "%r10")
    Push(addr.base_);
  VisitExpr(assign->rhs_);
  if (addr.base_ == "%r10")
    Pop(addr.base_);

  if (assign->Type()->IsScalar()) {
      EmitStore(addr, assign->Type());
  } else {
    // struct/union type
    // The address of rhs is in %rax
    CopyStruct(addr, assign->Type()->Width());
  }
}


void Generator::EmitStoreBitField(const ObjectAddr& addr, Type* type) {
  auto arithmType = type->ToArithm();
  assert(arithmType && arithmType->IsInteger());

  // The value to be stored is in %rax now
  auto mask = Object::BitFieldMask(addr.bitFieldBegin_, addr.bitFieldWidth_);

  Emit("salq", addr.bitFieldBegin_, "%rax");
  Emit("andq", mask, "%rax");
  Emit("movq", "%rax", "%r11");
  EmitLoad(addr.Repr(), arithmType);
  Emit("andq", ~mask, "%rax");
  Emit("orq", "%r11", "%rax");

  EmitStore(addr.Repr(), type);
}


void Generator::CopyStruct(ObjectAddr desAddr, int width) {
  int units[] = {8, 4, 2, 1};
  Emit("movq", "%rax", "%rcx");
  ObjectAddr srcAddr = {"", "%rcx", 0};
  for (auto unit: units) {
    while (width >= unit) {
      EmitLoad(srcAddr.Repr(), unit, false);
      EmitStore(desAddr.Repr(), unit, false);
      desAddr.offset_ += unit;
      srcAddr.offset_ += unit;
      width -= unit;
    }
  }
}


void Generator::GenCompOp(int width, bool flt, const char* set) {
  std::string cmp;
  if (flt) {
    cmp = width == 8 ? "ucomisd": "ucomiss";
  } else {
    cmp = GetInst("cmp", width, flt);
  }

  Emit(cmp, GetSrc(width, flt), GetDes(width, flt));
  Emit(set, "%al");
  Emit("movzbq", "%al", "%rax");
}


void Generator::GenDivOp(bool flt, bool sign, int width, int op) {
  if (flt) {
    auto inst = width == 4 ? "divss": "divsd";
    Emit(inst, "%xmm9", "%xmm0");
    return;
  }
  if (!sign) {
    Emit("xor", "%rdx", "%rdx");
    Emit(GetInst("div", width, flt), GetSrc(width, flt));
  } else {
    Emit(width == 4 ? "cltd": "cqto");
    Emit(GetInst("idiv", width, flt), GetSrc(width, flt));
  }
  if (op == '%')
    Emit("movq", "%rdx", "%rax");
}


void Generator::GenPointerArithm(BinaryOp* binary) {
  assert(binary->op_ == '+' || binary->op_ == '-');
  // For '+', we have swapped lhs_ and rhs_ to ensure that
  // the pointer is at lhs.
  Visit(binary->lhs_);
  Spill(false);
  Visit(binary->rhs_);
  Restore(false);

  auto type = binary->lhs_->Type()->ToPointer()->Derived();
  auto width = type->Width();
  if (binary->op_ == '+') {
    if (width > 1)
      Emit("imulq", width, "%r11");
    Emit("addq", "%r11", "%rax");
  } else {
    Emit("subq", "%r11", "%rax");
    if (width > 1) {
      Emit("movq", width, "%r11");
      GenDivOp(false, true, 8, '/');
    }
  }
}


// Only objects Allocated on stack
void Generator::VisitObject(Object* obj) {
  EmitLoc(obj);
  auto addr = LValGenerator().GenExpr(obj).Repr();

  if (!obj->Type()->IsScalar()) {
    // Return the address of the object in rax
    Emit("leaq", addr, "%rax");
  } else {
    EmitLoad(addr, obj->Type());
  }
}


void Generator::GenCastOp(UnaryOp* cast) {
  auto desType = cast->Type();
  auto srcType = cast->operand_->Type();

  if (srcType->IsFloat() && desType->IsFloat()) {
    if (srcType->Width() == desType->Width())
      return;
    auto inst = srcType->Width() == 4 ? "cvtss2sd": "cvtsd2ss";
    Emit(inst, "%xmm0", "%xmm0");
  } else if (srcType->IsFloat()) {
    // Handle bool
    if (desType->IsBool()) {
      Emit("pxor", "%xmm9", "%xmm9");
      GenCompOp(srcType->Width(), true, "setne");
    } else {
      auto inst = srcType->Width() == 4 ? "cvttss2si": "cvttsd2si";
      Emit(inst, "%xmm0", "%rax");
    }
  } else if (desType->IsFloat()) {
    auto inst = desType->Width() == 4 ? "cvtsi2ss": "cvtsi2sd";
    Emit(inst, "%rax", "%xmm0");
  } else if (srcType->ToPointer()
      || srcType->ToFunc()
      || srcType->ToArray()) {
    // Handle bool
    if (desType->IsBool()) {
      Emit("testq", "%rax", "%rax");
      Emit("setne", "%al");
    }
  } else {
    assert(srcType->ToArithm());
    int width = srcType->Width();
    auto sign = !srcType->IsUnsigned();
    const char* inst;
    switch (width) {
    case 1:
      inst = sign ? "movsbq": "movzbq";
      Emit(inst, GetReg(width), "%rax");
      break;
    case 2:
      inst = sign ? "movswq": "movzwq";
      Emit(inst, GetReg(width), "%rax");
      break;
    case 4: inst = "movl";
      if (desType->Width() == 8)
        Emit("cltq");
      break;
    case 8: break;
    }
    // Handle bool
    if (desType->IsBool()) {
      Emit("testq", "%rax", "%rax");
      Emit("setne", "%al");
    }
  }
}


void Generator::VisitUnaryOp(UnaryOp* unary) {
  EmitLoc(unary);
  switch  (unary->op_) {
  case Token::PREFIX_INC:
    return GenIncDec(unary->operand_, false, "add");
  case Token::PREFIX_DEC:
    return GenIncDec(unary->operand_, false, "sub");
  case Token::POSTFIX_INC:
    return GenIncDec(unary->operand_, true, "add");
  case Token::POSTFIX_DEC:
    return GenIncDec(unary->operand_, true, "sub");
  case Token::ADDR: {
    auto addr = LValGenerator().GenExpr(unary->operand_).Repr();
    Emit("leaq", addr, "%rax");
  } return;
  case Token::DEREF:
    return GenDerefOp(unary);
  case Token::PLUS:
    return VisitExpr(unary->operand_);
  case Token::MINUS:
    return GenMinusOp(unary);
  case '~':
    VisitExpr(unary->operand_);
    return Emit("notq", "%rax");
  case '!':
    VisitExpr(unary->operand_);
    GenCompZero(unary->operand_->Type());
    Emit("sete", "%al");
    Emit("movzbl", "%al", "%eax"); // Type of !operator is int
    return;
  case Token::CAST:
    Visit(unary->operand_);
    GenCastOp(unary);
    return;
  default: assert(false);
  }
}


void Generator::GenDerefOp(UnaryOp* deref) {
  VisitExpr(deref->operand_);
  if (deref->Type()->IsScalar()) {
    ObjectAddr addr {"", "%rax", 0};
    EmitLoad(addr.Repr(), deref->Type());
  } else {
    // Just let it go!
  }
}


void Generator::GenMinusOp(UnaryOp* minus) {
  auto width = minus->Type()->Width();
  auto flt = minus->Type()->IsFloat();

  VisitExpr(minus->operand_);

  if (flt) {
    Emit("pxor", "%xmm9", "%xmm9");
    Emit(GetInst("sub", width, flt), "%xmm0", "%xmm9");
    Emit(GetInst("mov", width, flt), "%xmm9", "%xmm0");
  } else {
    Emit(GetInst("neg", width, flt), GetDes(width, flt));
  }
}


void Generator::GenIncDec(Expr* operand,
                          bool postfix,
                          const std::string& inst) {
  auto width = operand->Type()->Width();
  auto flt = operand->Type()->IsFloat();

  auto addr = LValGenerator().GenExpr(operand).Repr();
  EmitLoad(addr, operand->Type());
  if (postfix) Save(flt);

  Constant* cons;
  auto pointerType = operand->Type()->ToPointer();
   if (pointerType) {
    long width = pointerType->Derived()->Width();
    cons = Constant::New(operand->Tok(), T_LONG, width);
  } else if (operand->Type()->IsInteger()) {
    cons = Constant::New(operand->Tok(), T_LONG, 1L);
  } else {
    if (width == 4)
      cons = Constant::New(operand->Tok(), T_FLOAT, 1.0f);
    else
      cons = Constant::New(operand->Tok(), T_DOUBLE, 1.0);
  }

  Emit(GetInst(inst, operand->Type()), ConsLabel(cons), GetDes(width, flt));
  EmitStore(addr, operand->Type());
  if (postfix && flt) {
    Emit("movsd", "%xmm9", "%xmm0");
  } else if (postfix) {
    Emit("mov", "%r11", "%rax");
  }
}


void Generator::VisitConditionalOp(ConditionalOp* condOp) {
  EmitLoc(condOp);
  auto ifStmt = IfStmt::New(condOp->cond_,
      condOp->exprTrue_, condOp->exprFalse_);
  VisitIfStmt(ifStmt);
}


void Generator::VisitEnumerator(Enumerator* enumer) {
  EmitLoc(enumer);
  auto cons = Constant::New(enumer->Tok(), T_INT, (long)enumer->Val());
  Visit(cons);
}


// Ident must be function
void Generator::VisitIdentifier(Identifier* ident) {
  EmitLoc(ident);
  Emit("leaq", ident->Name(), "%rax");
}


void Generator::VisitConstant(Constant* cons) {
  EmitLoc(cons);
  auto label = ConsLabel(cons);

  if (!cons->Type()->IsScalar()) {
    Emit("leaq", label, "%rax");
  } else {
    auto width = cons->Type()->Width();
    auto flt = cons->Type()->IsFloat();
    auto load = GetInst("mov", width, flt);
    auto des = GetDes(width, flt);
    Emit(load, label, des);
  }
}


// Use %ecx as temp register
// TempVar is only used for condition expression of 'switch'
// and struct copy
void Generator::VisitTempVar(TempVar* tempVar) {
  assert(tempVar->Type()->IsInteger());
  Emit("movl", "%ecx", "%eax");
}


void Generator::VisitDeclaration(Declaration* decl) {
  EmitLoc(decl->obj_);
  auto obj = decl->obj_;

  if (!obj->IsStatic()) {
    // The object has no linkage and has
    // no static storage(the object is on stack).
    // If it has no initialization,
    // then it's value is random initialized.
    if (!obj->HasInit())
      return;

    int lastEnd = obj->Offset();
    for (const auto& init: decl->Inits()) {
      ObjectAddr addr = ObjectAddr(obj->Offset() + init.offset_);
      addr.bitFieldBegin_ = init.bitFieldBegin_;
      addr.bitFieldWidth_ = init.bitFieldWidth_;
      if (lastEnd != addr.offset_)
        EmitZero(ObjectAddr(lastEnd), addr.offset_ - lastEnd);
      VisitExpr(init.expr_);
      if (init.type_->IsScalar()) {
        EmitStore(addr, init.type_);
      } else if (init.type_->ToStruct()) {
        CopyStruct(addr, init.type_->Width());
      } else {
        assert(false);
      }
      lastEnd = addr.offset_ + init.type_->Width();
    }
    auto objEnd = obj->Offset() + obj->Type()->Width();
    if (lastEnd != objEnd)
      EmitZero(ObjectAddr(lastEnd), objEnd - lastEnd);
    return;
  }

  if (obj->Linkage() == L_NONE)
    staticDecls_.push_back(decl);
  else
    GenStaticDecl(decl);
}


void Generator::GenStaticDecl(Declaration* decl) {
  auto obj = decl->obj_;
  assert(obj->IsStatic());

  const auto& label = obj->Repr();
  const auto width = obj->Type()->Width();
  const auto align = obj->Align();

  // Omit the external without initilizer
  if ((obj->Storage() & S_EXTERN) && !obj->HasInit())
    return;

  Emit(".data");
  auto glb = obj->Linkage() == L_EXTERNAL ? ".globl": ".local";
  Emit(glb, label);

  if (!obj->HasInit()) {
    Emit(".comm", label + ", " +  std::to_string(width) +
                  ", " + std::to_string(align));
    return;
  }

  Emit(".align", std::to_string(align));
  Emit(".type", label, "@object");
  // Does not decide the size of obj
  Emit(".size", label, std::to_string(width));
  EmitLabel(label);

  int offset = 0;
  auto iter = decl->Inits().begin();
  for (; iter != decl->Inits().end();) {
    auto staticInit = GetStaticInit(iter,
        decl->Inits().end(), std::max(iter->offset_, offset));

    if (staticInit.offset_ > offset)
      Emit(".zero", std::to_string(staticInit.offset_ - offset));

    switch (staticInit.width_) {
    case 1:
      Emit(".byte", std::to_string(static_cast<char>(staticInit.val_)));
      break;
    case 2:
      Emit(".value", std::to_string(static_cast<short>(staticInit.val_)));
      break;
    case 4:
      Emit(".long", std::to_string(static_cast<int>(staticInit.val_)));
      break;
    case 8: {
      std::string val;
      if (staticInit.label_.size() == 0) {
        val = std::to_string(staticInit.val_);
      } else if (staticInit.val_ != 0) {
        val = staticInit.label_ + "+" + std::to_string(staticInit.val_);
      } else {
        val = staticInit.label_;
      }
      Emit(".quad", val);
    } break;
    default: assert(false);
    }
    offset = staticInit.offset_ + staticInit.width_;
  }
  // Decides the size of object
  if (width > offset)
    Emit(".zero", std::to_string(width - offset));
}


void Generator::VisitEmptyStmt(EmptyStmt* emptyStmt) {
  assert(false);
}


void Generator::VisitIfStmt(IfStmt* ifStmt) {
  VisitExpr(ifStmt->cond_);

  // Compare to 0
  auto elseLabel = LabelStmt::New();
  auto endLabel = LabelStmt::New();

  GenCompZero(ifStmt->cond_->Type());

  if (ifStmt->else_) {
    Emit("je", elseLabel);
  } else {
    Emit("je", endLabel);
  }

  VisitStmt(ifStmt->then_);

  if (ifStmt->else_) {
    Emit("jmp", endLabel);
    EmitLabel(elseLabel->Repr());
    VisitStmt(ifStmt->else_);
  }

  EmitLabel(endLabel->Repr());
}


void Generator::VisitJumpStmt(JumpStmt* jumpStmt) {
  Emit("jmp", jumpStmt->label_);
}


void Generator::VisitLabelStmt(LabelStmt* labelStmt) {
  EmitLabel(labelStmt->Repr());
}


void Generator::VisitReturnStmt(ReturnStmt* returnStmt) {
  auto expr = returnStmt->expr_;
  if (expr) { // The return expr could be nil
    Visit(expr);
    if (expr->Type()->ToStruct()) {
      // %rax now has the address of the struct/union
      ObjectAddr addr = ObjectAddr(retAddrOffset_);
      Emit("movq", addr, "%r11");
      addr = {"", "%r11", 0};
      CopyStruct(addr, expr->Type()->Width());
      Emit("movq", "%r11", "%rax");
    }
  }
  Emit("jmp", curFunc_->retLabel_);
}


class Comp {
public:
  bool operator()(Object* lhs, Object* rhs) {
    return lhs->Align() < rhs->Align();
  }
};


void Generator::AllocObjects(Scope* scope, const FuncDef::ParamList& params) {
  int offset = offset_;

  auto paramSet = std::set<Object*>(params.begin(), params.end());
  std::priority_queue<Object*, std::vector<Object*>, Comp> heap;
  for (auto iter = scope->begin(); iter != scope->end(); ++iter) {
    auto obj = iter->second->ToObject();
    if (!obj || obj->IsStatic())
      continue;
    if (paramSet.find(obj) != paramSet.end())
      continue;
    heap.push(obj);
  }

  while (!heap.empty()) {
    auto obj = heap.top();
    heap.pop();

    offset -= obj->Type()->Width();
    auto align = obj->Align();
    if (obj->Type()->ToArray()) {
      // The alignment of an array is at least the aligment of a pointer
      // (as it is always cast to a pointer)
      align = std::min(align, 8);
    }
    offset = Type::MakeAlign(offset, align);
    obj->SetOffset(offset);
  }

  offset_ = offset;
}


void Generator::VisitCompoundStmt(CompoundStmt* compStmt) {
  if (compStmt->scope_) {
    AllocObjects(compStmt->scope_);
  }

  for (auto stmt: compStmt->stmts_) {
    Visit(stmt);
  }
}


void Generator::GetParamRegOffsets(int& gpOffset,
                                   int& fpOffset,
                                   int& overflow,
                                   FuncType* funcType) {
  TypeList types;
  for (auto param: funcType->Params())
    types.push_back(param->Type());
  auto locations = GetParamLocations(types, funcType->Derived());
  gpOffset = 0;
  fpOffset = 48;
  overflow = 16;
  for (const auto& loc: locations.locs_) {
    if (loc[1] == 'x')
      fpOffset += 16;
    else if (loc[1] == 'm')
      overflow += 8;
    else
      gpOffset += 8;
  }
}


void Generator::GenBuiltin(FuncCall* funcCall) {
  struct va_list_imp {
    unsigned int gp_offset;
    unsigned int fp_offset;
    void *overflow_arg_area;
    void *reg_save_area;
  };

  auto ap = UnaryOp::New(Token::DEREF, funcCall->args_[0]);
  auto addr = LValGenerator().GenExpr(ap);
  auto type = funcCall->FuncType();

  auto offset = offsetof(va_list_imp, reg_save_area);
  addr.offset_ += offset;
  const auto& saveAreaAddr = addr.Repr();
  addr.offset_ -= offset;

  offset = offsetof(va_list_imp, overflow_arg_area);
  addr.offset_ += offset;
  const auto& overflowAddr = addr.Repr();
  addr.offset_ -= offset;

  offset = offsetof(va_list_imp, gp_offset);
  addr.offset_ += offset;
  const auto& gpOffsetAddr = addr.Repr();
  addr.offset_ -= offset;

  offset = offsetof(va_list_imp, fp_offset);
  addr.offset_ += offset;
  const auto& fpOffsetAddr = addr.Repr();
  addr.offset_ -= offset;

  if (type == Parser::vaStartType_) {
    Emit("leaq", "-176(%rbp)", "%rax");
    Emit("movq", "%rax", saveAreaAddr);

    int gpOffset, fpOffset, overflowOffset;
    GetParamRegOffsets(gpOffset, fpOffset,
                       overflowOffset, curFunc_->FuncType());
    Emit("leaq", ObjectAddr(overflowOffset), "%rax");
    Emit("movq", "%rax", overflowAddr);
    Emit("movl", gpOffset, "%eax");
    Emit("movl", "%eax", gpOffsetAddr);
    Emit("movl", fpOffset, "%eax");
    Emit("movl", "%eax", fpOffsetAddr);
  } else if (type == Parser::vaArgType_) {
    static int cnt[2] = {0, 0};
    auto overflowLabel = ".L_va_arg_overflow" + std::to_string(++cnt[0]);
    auto endLabel = ".L_va_arg_end" + std::to_string(++cnt[1]);

    auto argType = funcCall->args_[1]->Type()->ToPointer()->Derived();
    auto cls = Classify(argType.GetPtr());
    if (cls == ParamClass::INTEGER) {
      Emit("movq", saveAreaAddr, "%rax");
      Emit("movq", "%rax", "%r11");
      Emit("movl", gpOffsetAddr, "%eax");
      Emit("cltq");
      Emit("cmpq", 48, "%rax");
      Emit("jae",  overflowLabel);
      Emit("addq", "%rax", "%r11");
      Emit("addq", 8, "%rax");
      Emit("movl", "%eax", gpOffsetAddr);
      Emit("movq", "%r11", "%rax");
      Emit("jmp",  endLabel);
    } else if (cls == ParamClass::SSE) {
      Emit("movq", saveAreaAddr, "%rax");
      Emit("movq", "%rax", "%r11");
      Emit("movl", fpOffsetAddr, "%eax");
      Emit("cltq");
      Emit("cmpq", 176, "%rax");
      Emit("jae",  overflowLabel);
      Emit("addq", "%rax", "%r11");
      Emit("addq", 16, "%rax");
      Emit("movl", "%eax", fpOffsetAddr);
      Emit("movq", "%r11", "%rax");
      Emit("jmp",  endLabel);
    } else if (cls == ParamClass::MEMORY) {
    } else {
      Error("internal error");
    }
    EmitLabel(overflowLabel);
    Emit("movq", overflowAddr, "%rax");
    Emit("movq", "%rax", "%r11");
    // Arguments passed by memory is aligned by at least 8 bytes
    Emit("addq", Type::MakeAlign(argType->Width(), 8), "%r11");
    Emit("movq", "%r11", overflowAddr);
    EmitLabel(endLabel);
  } else {
    assert(false);
  }
}


void Generator::VisitFuncCall(FuncCall* funcCall) {
  EmitLoc(funcCall);
  auto funcType = funcCall->FuncType();
  if (Parser::IsBuiltin(funcType))
    return GenBuiltin(funcCall);

  auto base = offset_;
  // Alloc memory for return value if it is struct/union
  int retStructOffset;
  auto retType = funcCall->Type()->ToStruct();
  if (retType) {
    retStructOffset = offset_;
    retStructOffset -= retType->Width();
    retStructOffset = Type::MakeAlign(retStructOffset, retType->Align());
    // No!!! you can't suppose that the
    // visition of arguments won't change the value of %rdi
    //Emit("leaq %d(#rbp), #rdi", offset);
    offset_ = retStructOffset;
  }

  TypeList types;
  for (auto arg: funcCall->args_) {
    types.push_back(arg->Type());
  }

  const auto& locations = GetParamLocations(types, retType);
  // Align stack frame by 16 bytes
  const auto& locs = locations.locs_;
  auto byMemCnt = locs.size() - locations.regCnt_ - locations.xregCnt_;

  offset_ = Type::MakeAlign(offset_ - byMemCnt * 8, 16) + byMemCnt * 8;
  for (int i = locs.size() - 1; i >=0; --i) {
    if (locs[i][1] == 'm') {
      Visit(funcCall->args_[i]);
      Push(funcCall->args_[i]->Type());
    }
  }

  for (int i = locs.size() - 1; i >= 0; --i) {
    if (locs[i][1] == 'm')
      continue;
    Visit(funcCall->args_[i]);
    Push(funcCall->args_[i]->Type());
  }

  for (const auto& loc: locs) {
    if (loc[1] != 'm')
      Pop(loc);
  }

  // If variadic, set %al to floating param number
  if (funcType->Variadic()) {
    Emit("movq", locations.xregCnt_, "%rax");
  }
  if (retType) {
    Emit("leaq", ObjectAddr(retStructOffset), "%rdi");
  }

  Emit("leaq", ObjectAddr(offset_), "%rsp");
  auto addr = LValGenerator().GenExpr(funcCall->Designator());
  if (addr.base_.size() == 0 && addr.offset_ == 0) {
    Emit("call", addr.label_);
  } else {
    Emit("leaq", addr, "%r10");
    Emit("call", "*%r10");
  }

  // Reset stack frame
  offset_ = base;
}


ParamLocations Generator::GetParamLocations(const TypeList& types,
                                            bool retStruct) {
  ParamLocations locations;

  locations.regCnt_ = retStruct;
  locations.xregCnt_ = 0;
  for (auto type: types) {
    auto cls = Classify(type);

    const char* reg = nullptr;
    if (cls == ParamClass::INTEGER) {
      if (locations.regCnt_ < regs.size())
        reg = regs[locations.regCnt_++];
    } else if (cls == ParamClass::SSE) {
      if (locations.xregCnt_ < xregs.size())
        reg = xregs[locations.xregCnt_++];
    }
    locations.locs_.push_back(reg ? reg: "%mem");
  }
  return locations;
}


void Generator::VisitFuncDef(FuncDef* funcDef) {
  curFunc_ = funcDef;

  auto name = funcDef->Name();

  Emit(".text");
  if (funcDef->Linkage() == L_INTERNAL) {
    Emit(".local", name);
  } else {
    Emit(".globl", name);
  }
  Emit(".type", name, "@function");

  EmitLabel(name);
  Emit("pushq", "%rbp");
  Emit("movq", "%rsp", "%rbp");

  offset_ = 0;

  auto& params = funcDef->FuncType()->Params();
  // Arrange space to store params passed by registers
  bool retStruct = funcDef->FuncType()->Derived()->ToStruct();
  TypeList types;
  for (auto param: params)
    types.push_back(param->Type());

  auto locations = GetParamLocations(types, retStruct);
  const auto& locs = locations.locs_;

  if (funcDef->FuncType()->Variadic()) {
    GenSaveArea(); // 'offset' is now the begin of save area
    if (retStruct) {
      retAddrOffset_ = offset_;
      offset_ += 8;
    }
    int regOffset = offset_;
    int xregOffset = offset_ + 48;
    int byMemOffset = 16;
    for (size_t i = 0; i < locs.size(); ++i) {
      if (locs[i][1] == 'm') {
        params[i]->SetOffset(byMemOffset);

        // TODO(wgtdkp): width of incomplete array ?
        // What about the var args, var args offset always increment by 8
        //byMemOffset += 8;
        byMemOffset += params[i]->Type()->Width();
        byMemOffset = Type::MakeAlign(byMemOffset, 8);
      } else if (locs[i][1] == 'x') {
        params[i]->SetOffset(xregOffset);
        xregOffset += 16;
      } else {
        params[i]->SetOffset(regOffset);
        regOffset += 8;
      }
    }
  } else {
    if (retStruct) {
      retAddrOffset_ = Push("%rdi");
    }
    int byMemOffset = 16;
    for (size_t i = 0; i < locs.size(); ++i) {
      if (locs[i][1] == 'm') {
        params[i]->SetOffset(byMemOffset);
        // TODO(wgtdkp): width of incomplete array ?
        byMemOffset += params[i]->Type()->Width();
        byMemOffset = Type::MakeAlign(byMemOffset, 8);
        continue;
      }
      params[i]->SetOffset(Push(locs[i]));
    }
  }

  AllocObjects(funcDef->Body()->Scope(), params);

  for (auto stmt: funcDef->body_->stmts_) {
    Visit(stmt);
  }

  EmitLabel(funcDef->retLabel_->Repr());
  Emit("leaveq");
  Emit("retq");
}


void Generator::GenSaveArea() {
  static const int begin = -176;
  int offset = begin;
  for (auto reg: regs) {
    Emit("movq", reg, ObjectAddr(offset));
    offset += 8;
  }
  Emit("testb", "%al", "%al");
  auto label = LabelStmt::New();
  Emit("je", label);
  for (auto xreg: xregs) {
    Emit("movaps", xreg, ObjectAddr(offset));
    offset += 16;
  }
  assert(offset == 0);
  EmitLabel(label->Repr());

  offset_ = begin;
}


void Generator::VisitTranslationUnit(TranslationUnit* unit) {
  for (auto extDecl: unit->ExtDecls()) {
    Visit(extDecl);

    // Float and string literal
    if (rodatas_.size())
      Emit(".section", ".rodata");
    for (auto rodata: rodatas_) {
      if (rodata.align_ == 1) { // Literal
        EmitLabel(rodata.label_);
        Emit(".string", "\"" + rodata.sval_ + "\"");
      } else if (rodata.align_ == 4) {
        Emit(".align", "4");
        EmitLabel(rodata.label_);
        Emit(".long", std::to_string(static_cast<int>(rodata.ival_)));
      } else {
        Emit(".align", "8");
        EmitLabel(rodata.label_);
        Emit(".quad", std::to_string(rodata.ival_));
      }
    }
    rodatas_.clear();

    for (auto staticDecl: staticDecls_) {
      GenStaticDecl(staticDecl);
    }
    staticDecls_.clear();
  }
}


void Generator::Gen() {
  Emit(".file", "\"" + filename_in + "\"");
  VisitTranslationUnit(parser_->Unit());
}


void Generator::EmitLoc(Expr* expr) {
  if (!debug) {
    return;
  }

  static int fileno = 0;
  if (expr->tok_ == nullptr) {
    return;
  }

  const auto loc = &expr->tok_->loc_;
  if (loc->filename_ != last_file) {
    Emit(".file", std::to_string(++fileno) + " \"" + *loc->filename_ + "\"");
    last_file = loc->filename_;
  }
  Emit(".loc", std::to_string(fileno) + " " +
               std::to_string(loc->line_) + " 0");

  std::string line;
  for (const char* p = loc->lineBegin_; *p && *p != '\n'; ++p)
    line.push_back(*p);
  Emit("# " + line);
}


void Generator::EmitLoad(const std::string& addr, Type* type) {
  assert(type->IsScalar());
  EmitLoad(addr, type->Width(), type->IsFloat());
}


void Generator::EmitLoad(const std::string& addr, int width, bool flt) {
  auto load = GetLoad(width, flt);
  auto des = GetDes(width == 4 ? 4: 8, flt);
  Emit(load, addr, des);
}


void Generator::EmitStore(const ObjectAddr& addr, Type* type) {
  if (addr.bitFieldWidth_ != 0) {
    EmitStoreBitField(addr, type);
  } else {
    EmitStore(addr.Repr(), type);
  }
}


void Generator::EmitStore(const std::string& addr, Type* type) {
  EmitStore(addr, type->Width(), type->IsFloat());
}


void Generator::EmitStore(const std::string& addr, int width, bool flt) {
  auto store = GetInst("mov", width, flt);
  auto des = GetDes(width, flt);
  Emit(store, des, addr);
}


void Generator::EmitLabel(const std::string& label) {
  fprintf(outFile_, "%s:\n", label.c_str());
}


void Generator::EmitZero(ObjectAddr addr, int width) {
  int units[] = {8, 4, 2, 1};
  Emit("xorq", "%rax", "%rax");
  for (auto unit: units) {
    while (width >= unit) {
      EmitStore(addr.Repr(), unit, false);
      addr.offset_ += unit;
      width -= unit;
    }
  }
}


void LValGenerator::VisitBinaryOp(BinaryOp* binary) {
  EmitLoc(binary);
  assert(binary->op_ == '.');

  addr_ = LValGenerator().GenExpr(binary->lhs_);
  const auto& name = binary->rhs_->Tok()->str_;
  auto structType = binary->lhs_->Type()->ToStruct();
  auto member = structType->GetMember(name);

  addr_.offset_ += member->Offset();
  addr_.bitFieldBegin_ = member->bitFieldBegin_;
  addr_.bitFieldWidth_ = member->bitFieldWidth_;
}


void LValGenerator::VisitUnaryOp(UnaryOp* unary) {
  EmitLoc(unary);
  assert(unary->op_ == Token::DEREF);
  Generator().VisitExpr(unary->operand_);
  Emit("movq", "%rax", "%r10");
  addr_ = {"", "%r10", 0};
}


void LValGenerator::VisitObject(Object* obj) {
  EmitLoc(obj);
  if (!obj->IsStatic() && obj->Anonymous()) {
    assert(obj->Decl());
    Generator().Visit(obj->Decl());
    obj->SetDecl(nullptr);
  }

  if (obj->IsStatic()) {
    addr_ = {obj->Repr(), "%rip", 0};
  } else {
    addr_ = {"", "%rbp", obj->Offset()};
  }
}


// The identifier must be function
void LValGenerator::VisitIdentifier(Identifier* ident) {
  assert(!ident->ToTypeName());
  EmitLoc(ident);
  // Function address
  addr_ = {ident->Name(), "", 0};
}


void LValGenerator::VisitTempVar(TempVar* tempVar) {
  std::string label;
  switch (tempVar->Type()->Width()) {
  case 1: label = "%cl"; break;
  case 2: label = "%cx"; break;
  case 4: label = "%ecx"; break;
  case 8: label = "%rcx"; break;
  default: assert(false);
  }
  addr_ = {label, "", 0};
}


std::string ObjectAddr::Repr() const {
  auto ret = base_.size() ? "(" + base_ + ")": "";
  if (label_.size() == 0) {
    if (offset_ == 0) {
      return ret;
    }
    return std::to_string(offset_) + ret;
  } else {
    if (offset_ == 0) {
      return label_ + ret;
    }
    return label_ + "+" + std::to_string(offset_) + ret;
  }
}


StaticInitializer Generator::GetStaticInit(InitList::iterator& iter,
                                           InitList::iterator end,
                                           int offset) {
  auto init = iter++;
  auto width = init->type_->Width();
  if (init->type_->IsInteger()) {
    if (init->bitFieldWidth_ == 0) {
      auto val = Evaluator<long>().Eval(init->expr_);
      return {init->offset_, width, val, ""};
    }
    int totalBits = 0;
    unsigned char val = 0;
    while (init != end && init->offset_ <= offset && totalBits < 8) {
      auto bitVal = Evaluator<long>().Eval(init->expr_);
      auto begin = init->bitFieldBegin_;
      auto width = init->bitFieldWidth_;
      auto valBegin = 0;
      auto valWidth = 0;
      auto mask = 0UL;
      if (init->offset_ < offset) {
        begin = 0;
        width -= (8 - init->bitFieldBegin_);
        if (offset - init->offset_ > 1)
          width -= (offset - init->offset_ - 1) * 8;
        valBegin = init->bitFieldWidth_ - width;
      }
      valWidth = std::min(static_cast<unsigned char>(8 - begin), width);
      mask = Object::BitFieldMask(valBegin, valWidth);
      val |= ((bitVal & mask) >> valBegin) << begin;
      totalBits = begin + valWidth;
      if (width - valWidth <= 0)
        ++init;
    }
    iter = init;
    return {offset, 1, val, ""};
  } else if (init->type_->IsFloat()) {
    auto val = Evaluator<double>().Eval(init->expr_);
    auto lval = *reinterpret_cast<long*>(&val);
    return {init->offset_, width, lval, ""};
  } else if (init->type_->ToPointer()) {
    auto addr = Evaluator<Addr>().Eval(init->expr_);
    return {init->offset_, width, addr.offset_, addr.label_};
  } else { // Struct initializer
    Error(init->expr_, "initializer element is not constant");
    return StaticInitializer(); // Make compiler happy
  }
}
