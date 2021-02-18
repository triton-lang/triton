#include "triton/lang/parser.h"

#include "triton/lang/cpp.h"
#include "triton/lang/encoding.h"
#include "triton/lang/error.h"
#include "triton/lang/evaluator.h"
#include "triton/lang/scope.h"
#include "triton/lang/type.h"

#include <iostream>
#include <set>
#include <string>
#include <climits>


FuncType* Parser::vaStartType_ {nullptr};
FuncType* Parser::vaArgType_ {nullptr};


FuncDef* Parser::EnterFunc(Identifier* ident) {
  curFunc_ = FuncDef::New(ident, LabelStmt::New());
  return curFunc_;
}


void Parser::ExitFunc() {
  // Resolve 那些待定的jump；
  // 如果有jump无法resolve，也就是有未定义的label，报错；
  for (auto iter = unresolvedJumps_.begin();
       iter != unresolvedJumps_.end(); ++iter) {
    auto label = iter->first;
    auto labelStmt = FindLabel(label->str_);
    if (labelStmt == nullptr) {
      Error(label, "label '%s' used but not defined",
          label->str_.c_str());
    }

    iter->second->SetLabel(labelStmt);
  }

  unresolvedJumps_.clear();	//清空未定的 jump 动作
  curLabels_.clear();	//清空 label map

  curFunc_ = nullptr;
}


void Parser::EnterBlock(FuncType* funcType) {
  curScope_ = new Scope(curScope_, S_BLOCK);
  if (funcType) {
    // Merge elements in param scope into current block scope
    for (auto param: funcType->Params())
      curScope_->Insert(param);
  }
}


void Parser::Parse() {
  DefineBuiltins();
  ParseTranslationUnit();
}


void Parser::ParseTranslationUnit() {
  while (!ts_.Peek()->IsEOF()) {
    if (ts_.Try(Token::STATIC_ASSERT)) {
      ParseStaticAssert();
      continue;
    } else if (ts_.Try(';')) {
      continue;
    }

    int storageSpec, funcSpec, align;
    auto declType = ParseDeclSpec(&storageSpec, &funcSpec, &align);
    auto declInfo = ParseDeclarator(declType);

    auto tok = declInfo.tok;
    auto type = declInfo.type;
    auto attrs = declInfo.attrs;

    if (tok == nullptr) {
      ts_.Expect(';');
      continue;
    }

    auto ident = ProcessDeclarator(tok, type, attrs, storageSpec, funcSpec, align);
    type = ident->Type();

    if (tok && type->ToFunc() && ts_.Try('{')) { // Function definition
      unit_->Add(ParseFuncDef(ident));
    } else { // Declaration
      auto decl = ParseInitDeclarator(ident);
      if (decl) unit_->Add(decl);

      while (ts_.Try(',')) {
        auto ident = ParseDirectDeclarator(declType, storageSpec,
                                           funcSpec, align);
        decl = ParseInitDeclarator(ident);
        if (decl) unit_->Add(decl);
      }
      // GNU extension: function/type/variable attributes
      TryAttributeSpecList();
      ts_.Expect(';');
    }
  }
}


FuncDef* Parser::ParseFuncDef(Identifier* ident) {
  auto funcDef = EnterFunc(ident);
  if (funcDef->FuncType()->Complete()) {
    Error(ident, "redefinition of '%s'", funcDef->Name().c_str());
  }
  // TODO(wgtdkp): param checking
  auto funcType = ident->Type()->ToFunc();
  funcType->SetComplete(true);
  for (auto param: funcType->Params()) {
    if (param->Anonymous())
      Error(param, "param name omitted");
  }
  funcDef->SetBody(ParseCompoundStmt(funcType));
  ExitFunc();

  return funcDef;
}


Expr* Parser::ParseExpr() {
  return ParseCommaExpr();
}


Expr* Parser::ParseCommaExpr() {
  auto lhs = ParseAssignExpr();
  auto tok = ts_.Peek();
  while (ts_.Try(',')) {
    auto rhs = ParseAssignExpr();
    lhs = BinaryOp::New(tok, lhs, rhs);

    tok = ts_.Peek();
  }
  return lhs;
}


Expr* Parser::ParsePrimaryExpr() {
  if (ts_.Empty()) {
    Error(ts_.Peek(), "premature end of input");
  }

  auto tok = ts_.Next();
  if (tok->tag_ == '(') {
    auto expr = ParseExpr();
    ts_.Expect(')');
    return expr;
  }

  if (tok->IsIdentifier()) {
    auto ident = curScope_->Find(tok);
    if (ident) return ident;
    if (IsBuiltin(tok->str_)) return GetBuiltin(tok);
    Error(tok, "undefined symbol '%s'", tok->str_.c_str());
  } else if (tok->IsConstant()) {
    return ParseConstant(tok);
  } else if (tok->IsLiteral()) {
    return ConcatLiterals(tok);
  } else if (tok->tag_ == Token::GENERIC) {
    return ParseGeneric();
  }

  Error(tok, "'%s' unexpected", tok->str_.c_str());
  return nullptr; // Make compiler happy
}


static void ConvertLiteral(std::string& val, Encoding enc) {
  switch (enc) {
  case Encoding::NONE:
  case Encoding::UTF8: break;
  case Encoding::CHAR16: ConvertToUTF16(val); break;
  case Encoding::CHAR32:
  case Encoding::WCHAR: ConvertToUTF32(val); break;
  }
}


Constant* Parser::ConcatLiterals(const Token* tok) {
  auto val = new std::string;
  auto enc = Scanner(tok).ScanLiteral(*val);
  ConvertLiteral(*val, enc);
  while (ts_.Test(Token::LITERAL)) {
    auto nextTok = ts_.Next();
    std::string nextVal;
    auto nextEnc = Scanner(nextTok).ScanLiteral(nextVal);
    ConvertLiteral(nextVal, nextEnc);
    if (enc == Encoding::NONE) {
      ConvertLiteral(*val, nextEnc);
      enc = nextEnc;
    }
    if (nextEnc != Encoding::NONE && nextEnc != enc)
      Error(nextTok, "cannot concat lietrals with different encodings");
    *val += nextVal;
  }

  int tag = T_CHAR;
  switch (enc) {
  case Encoding::NONE:
  case Encoding::UTF8:
    tag = T_CHAR; val->append(1, '\0'); break;
  case Encoding::CHAR16:
    tag = T_UNSIGNED | T_SHORT; val->append(2, '\0'); break;
  case Encoding::CHAR32:
  case Encoding::WCHAR:
    tag = T_UNSIGNED | T_INT; val->append(4, '\0'); break;
  }

  return Constant::New(tok, tag, val);
}


Encoding Parser::ParseLiteral(std::string& str, const Token* tok) {
  return Scanner(tok).ScanLiteral(str);
}


Constant* Parser::ParseConstant(const Token* tok) {
  assert(tok->IsConstant());
  if (tok->tag_ == Token::I_CONSTANT) {
    return ParseInteger(tok);
  } else if (tok->tag_ == Token::C_CONSTANT) {
    return ParseCharacter(tok);
  } else {
    return ParseFloat(tok);
  }
}


Constant* Parser::ParseFloat(const Token* tok) {
  const auto& str = tok->str_;
  size_t end = 0;
  double val = 0.0;
  try {
    val = stod(str, &end);
  } catch (const std::out_of_range& oor) {
    Error(tok, "float out of range");
  }

  int tag = T_DOUBLE;
  if (str[end] == 'f' || str[end] == 'F') {
    tag = T_FLOAT;
    ++end;
  } else if (str[end] == 'l' || str[end] == 'L') {
    tag = T_LONG | T_DOUBLE;
    ++end;
  }
  if (str[end] != 0)
    Error(tok, "invalid suffix");
  return Constant::New(tok, tag, val);
}


Constant* Parser::ParseCharacter(const Token* tok) {
  int val;
  auto enc = Scanner(tok).ScanCharacter(val);

  int tag;
  switch (enc) {
  case Encoding::NONE:
    val = (char)val;
    tag = T_INT; break;
  case Encoding::CHAR16:
    val = (char16_t)val;
    tag = T_UNSIGNED | T_SHORT; break;
  case Encoding::WCHAR:
  case Encoding::CHAR32: tag = T_UNSIGNED | T_INT; break;
  default: assert(false);
  }
  return Constant::New(tok, tag, static_cast<long>(val));
}


Constant* Parser::ParseInteger(const Token* tok) {
  const auto& str = tok->str_;
  size_t end = 0;
  long val = 0;
  try {
    val = stoull(str, &end, 0);
  } catch (const std::out_of_range& oor) {
    Error(tok, "integer out of range");
  }

  int tag = 0;
  for (; str[end]; ++end) {
    if (str[end] == 'u' || str[end] == 'U') {
      if (tag & T_UNSIGNED)
        Error(tok, "invalid suffix");
      tag |= T_UNSIGNED;
    } else {
      if ((tag & T_LONG) || (tag & T_LLONG))
        Error(tok, "invalid suffix");
      if (str[end + 1] == 'l' || str[end + 1] =='L') {
        tag |= T_LLONG;
        ++end;
      } else {
        tag |= T_LONG;
      }
    }
  }

  bool decimal = ('1' <= str[0] && str[0] <= '9');
  if (decimal) {
    switch (tag) {
    case 0:
      tag |= !(val & ~(long)INT_MAX) ? T_INT: T_LONG; break;
    case T_UNSIGNED:
      tag |= !(val & ~(long)UINT_MAX) ? T_INT: T_LONG; break;
    case T_LONG: break;
    case T_UNSIGNED | T_LONG: break;
    }
  } else {
    switch (tag) {
    case 0:
      tag |= !(val & ~(long)INT_MAX) ? T_INT
           : !(val & ~(long)UINT_MAX) ? T_UNSIGNED
           : !(val & ~(long)LONG_MAX) ? T_LONG
           : T_UNSIGNED | T_LONG; break;
    case T_UNSIGNED:
      tag |= !(val & ~(long)UINT_MAX) ? T_INT: T_LONG; break;
    case T_LONG:
      tag |= !(val & ~(long)LONG_MAX) ? 0: T_UNSIGNED; break;
    case T_UNSIGNED | T_LONG:
      break;
    }
  }

  return Constant::New(tok, tag, val);
}


Expr* Parser::ParseGeneric() {
  ts_.Expect('(');
  auto controlExpr = ParseAssignExpr();
  ts_.Expect(',');
  Expr* selectedExpr = nullptr;
  bool isDefault = false;
  while (true) {
    if (ts_.Try(Token::DEFAULT)) {
      ts_.Expect(':');
      auto defaultExpr = ParseAssignExpr();
      if (!selectedExpr) {
        selectedExpr = defaultExpr;
        isDefault = true;
      }
    } else {
      auto tok = ts_.Peek();
      auto type = ParseTypeName();
      ts_.Expect(':');
      auto expr = ParseAssignExpr();
      if (type->Compatible(*controlExpr->Type())) {
        if (selectedExpr && !isDefault) {
          Error(tok, "more than one generic association"
              " are compatible with control expression");
        }
        selectedExpr = expr;
        isDefault = false;
      }
    }
    if (!ts_.Try(',')) {
      ts_.Expect(')');
      break;
    }
  }

  if (!selectedExpr)
    Error(ts_.Peek(), "no compatible generic association");
  return selectedExpr;
}


QualType Parser::TryCompoundLiteral() {
  auto mark = ts_.Mark();
  if (ts_.Try('(') && IsTypeName(ts_.Peek())) {
    auto type = ParseTypeName();
    if (ts_.Try(')') && ts_.Test('{'))
      return type;
  }
  ts_.ResetTo(mark);
  return nullptr;
}


Expr* Parser::ParsePostfixExpr() {
  if (ts_.Peek()->IsEOF()) {
    Error(ts_.Peek(), "premature end of input");
  }

  auto type = TryCompoundLiteral();
  if (type) {
    auto anony = ParseCompoundLiteral(type);
    return ParsePostfixExprTail(anony);
  }

  Expr* primExpr;
  //FIXME: merge into generic array functions
  if(ts_.Try(Token::EXP))
    primExpr = ParseUnaryIntrinsicOp(Token::EXP);
  else if(ts_.Try(Token::SQRTF))
    primExpr = ParseUnaryIntrinsicOp(Token::SQRTF);
  else if(ts_.Try(Token::LOG))
    primExpr = ParseUnaryIntrinsicOp(Token::LOG);
  else
    primExpr = ParsePrimaryExpr();
  return ParsePostfixExprTail(primExpr);
}


Object* Parser::ParseCompoundLiteral(QualType type) {
  auto linkage = curScope_->Type() == S_FILE ? L_INTERNAL: L_NONE;
  auto anony = Object::NewAnony(ts_.Peek(), type, 0, linkage);
  auto decl = ParseInitDeclaratorSub(anony);

  // Just for generator to find the compound literal
  if (curScope_->Type() == S_FILE) {
    unit_->Add(decl);
  } else {
    curScope_->Insert(anony->Repr(), anony);
  }
  return anony;
}


// Return the constructed postfix expression
Expr* Parser::ParsePostfixExprTail(Expr* lhs) {
  while (true) {
    auto tok = ts_.Next();

    switch (tok->tag_) {
    case '[': lhs = ParseSubScripting(lhs); break;
    case '(': lhs = ParseFuncCall(lhs); break;
    case Token::PTR: lhs = UnaryOp::New(Token::DEREF, lhs);
    // Fall through
    case '.': lhs = ParseMemberRef(tok, '.', lhs); break;
    case Token::INC:
    case Token::DEC: lhs = ParsePostfixIncDec(tok, lhs); break;
    default: ts_.PutBack(); return lhs;
    }
  }
}


Expr* Parser::ParseSubScripting(Expr* lhs) {
  auto lhsTile = lhs->Type()->ToTile();
  if(lhsTile == nullptr)
    Error(lhs, "tile expected");
  TileType::ShapeInt lhsShape = lhsTile->Shape();
  QualType lhsQual = lhsTile->Derived();
  // create ret shape
  TileType::ShapeInt shape;
  TileType::ShapeInt axVec;
  size_t i = 0;
  const Token* tok;
  std::vector<std::pair<int, int>> redInfo;
  do {
    tok = ts_.Next();
    switch(tok->tag_) {
      case ':':
        shape.push_back(lhsShape[i++]);
        break;
      case Token::NEWAXIS:
        shape.push_back(1);
        break;
      case Token::ADD:
      case Token::SUB:
      case Token::MAX:
      case Token::MIN:{
        int info = UnaryOp::encodeRed(i, tok->tag_);
        redInfo.push_back({i, info});
        shape.push_back(lhsShape[i++]);
        break;
      }
      case '^':{
        Expr* expr = ParseConditionalExpr();
        EnsureInteger(expr);
        int ax = Evaluator<long>().Eval(expr);
        axVec.push_back(ax);
        if(ax < 0 || ax >= lhsShape.size())
          Error(tok, "unknown axis %d in transposition", ax);
        shape.push_back(lhsShape[ax]);
        i++;
        break;
      }

      default:
        Error(tok, "Unexpected subscript symbol encountered at dimension %d", i);
        break;
    }
  }while(ts_.Try(','));
  ts_.Expect(']');

  // transposition mode
  std::set<int> axSet(axVec.begin(), axVec.end());
  if(!axSet.empty()){
    if(axSet.size()!=lhsShape.size())
      Error(tok, "transposition must address all axes of input array");
    return TransOp::New(axVec, lhs);
  }

  // broadcasting mode
  if(lhsShape.size() > i)
    Error(tok, "broadcasting not using all operand axes");

  // create ret tile
  Expr* res = lhs;
  for(auto r: redInfo){
    shape.erase(shape.begin() + r.first);
    Type *retType;
    if(shape.empty())
      retType = lhsQual.GetPtr();
    else
      retType = TileType::New(shape, lhsQual);
    res = UnaryOp::New(Token::REDUCE, res, retType, r.second);
  }
  if(!shape.empty()){
    TileType *retType = TileType::New(shape, lhsQual);
    res = UnaryOp::New(Token::CAST, res, retType);
  }
  return res;
}


BinaryOp* Parser::ParseMemberRef(const Token* tok, int op, Expr* lhs) {
  auto memberName = ts_.Peek()->str_;
  ts_.Expect(Token::IDENTIFIER);

  auto structUnionType = lhs->Type()->ToStruct();
  if (structUnionType == nullptr) {
    Error(tok, "an struct/union expected");
  }

  auto rhs = structUnionType->GetMember(memberName);
  if (rhs == nullptr) {
    Error(tok, "'%s' is not a member of '%s'",
        memberName.c_str(), "[obj]");
  }

  return  BinaryOp::New(tok, op, lhs, rhs);
}


UnaryOp* Parser::ParsePostfixIncDec(const Token* tok, Expr* operand) {
  auto op = tok->tag_ == Token::INC ?
            Token::POSTFIX_INC: Token::POSTFIX_DEC;
  return UnaryOp::New(op, operand);
}


FuncCall* Parser::ParseFuncCall(Expr* designator) {
  FuncCall::ArgList args;
  while (!ts_.Try(')')) {
    args.push_back(Expr::MayCast(ParseAssignExpr()));
    if (!ts_.Test(')'))
      ts_.Expect(',');
  }
  return FuncCall::New(designator, args);
}


Expr* Parser::ParseUnaryExpr() {
  auto tok = ts_.Next();
  switch (tok->tag_) {
  case Token::ALIGNOF: return ParseAlignof();
  case Token::SIZEOF: return ParseSizeof();
  case Token::INC: return ParsePrefixIncDec(tok);
  case Token::DEC: return ParsePrefixIncDec(tok);
  case '&': return ParseUnaryOp(tok, Token::ADDR);
  case '*': return ParseDerefOp(tok);
  case '+': return ParseUnaryOp(tok, Token::PLUS);
  case '-': return ParseUnaryOp(tok, Token::MINUS);
  case '~': return ParseUnaryOp(tok, '~');
  case '!': return ParseUnaryOp(tok, '!');
  case '^': {
    auto operand = ParseCastExpr();
    TileType::ShapeInt shape = operand->Type()->ToTile()->Shape();
    TransOp::PermInt perm(shape.size());
    for(int d = 0; d < shape.size(); d++)
      perm[d] = d;
    std::rotate(perm.begin(), perm.begin() + 1, perm.end());
    return TransOp::New(perm, operand);
  }
  default:
    ts_.PutBack();
    return ParsePostfixExpr();
  }
}


Constant* Parser::ParseSizeof() {
  QualType type(nullptr);
  auto tok = ts_.Next();
  if (tok->tag_ == '(' && IsTypeName(ts_.Peek())) {
    type = ParseTypeName();
    ts_.Expect(')');
  } else {
    ts_.PutBack();
    auto expr = ParseUnaryExpr();
    type = expr->Type();
  }

  if (type->ToFunc() || type->ToVoid()) {
  } else if (!type->Complete()) {
    Error(tok, "sizeof(incomplete type)");
  }
  long val = type->Width();
  return Constant::New(tok, T_UNSIGNED | T_LONG, val);
}


Constant* Parser::ParseAlignof() {
  ts_.Expect('(');
  auto tok = ts_.Peek();
  auto type = ParseTypeName();
  ts_.Expect(')');

  long val = type->Align();
  return Constant::New(tok, T_UNSIGNED| T_LONG, val);
}


UnaryOp* Parser::ParsePrefixIncDec(const Token* tok) {
  assert(tok->tag_ == Token::INC || tok->tag_ == Token::DEC);

  auto op = tok->tag_ == Token::INC ?
            Token::PREFIX_INC: Token::PREFIX_DEC;
  auto operand = ParseUnaryExpr();
  return UnaryOp::New(op, operand);
}

UnaryOp* Parser::ParseUnaryIntrinsicOp(int op) {
  ts_.Expect('(');
  auto operand = ParseExpr();
  ts_.Expect(')');
  auto ret = UnaryOp::New(op, operand);
  return ret;
}

UnaryOp* Parser::ParseUnaryOp(const Token* tok, int op) {
  auto operand = ParseCastExpr();
  return UnaryOp::New(op, operand);
}

Expr* Parser::ParseDerefOp(const Token* tok) {
  Expr* pred = nullptr;
  if(ts_.Try('?')){
    ts_.Expect('(');
    pred = ParseExpr();
    ts_.Expect(')');
  }
  Expr* addr = ParseCastExpr();
  if(pred)
    return BinaryOp::New(tok, Token::MASKED_DEREF, pred, addr);
  else
    return UnaryOp::New(Token::DEREF, addr);
}

QualType Parser::ParseTypeName() {
  auto type = ParseSpecQual();
  if (ts_.Test('*') || ts_.Test('(') || ts_.Test('[')) // abstract-declarator FIRST set
    return ParseAbstractDeclarator(type);
  return type;
}


Expr* Parser::ParseCastExpr() {
  auto tok = ts_.Next();
  // bitcast
  if (tok->tag_ == Token::BITCAST) {
    ts_.Expect('<');
    auto type = ParseTypeName();
    ts_.Expect('>');
    ts_.Expect('(');
    auto operand = ParseExpr();
    ts_.Expect(')');
    return UnaryOp::New(Token::BITCAST, operand, type);
  }
  // semantic cast
  if (tok->tag_ == '(' && IsTypeName(ts_.Peek())) {
    auto type = ParseTypeName();
    ts_.Expect(')');
    if (ts_.Test('{')) {
      auto anony = ParseCompoundLiteral(type);
      return ParsePostfixExprTail(anony);
    }
    auto operand = ParseCastExpr();
    return UnaryOp::New(Token::CAST, operand, type);
  }

  ts_.PutBack();
  return ParseUnaryExpr();
}

Expr* Parser::ParseRangeExpr() {
  auto lhs = ParseCastExpr();
  auto tok = ts_.Next();
  while (tok->tag_ == Token::ELLIPSIS) {
    auto rhs = ParseCastExpr();
    lhs = BinaryOp::New(tok, lhs, rhs);
    tok = ts_.Next();
  }
  ts_.PutBack();
  return  lhs;
}

Expr* Parser::ParseMatmulExpr() {
  auto lhs = ParseRangeExpr();
  auto tok = ts_.Next();
  while (tok->tag_ == Token::MATMUL) {
    auto rhs = ParseRangeExpr();
    lhs = BinaryOp::New(tok, lhs, rhs);
    tok = ts_.Next();
  }
  ts_.PutBack();
  return lhs;
}

Expr* Parser::ParseMultiplicativeExpr() {
  auto lhs = ParseMatmulExpr();
  auto tok = ts_.Next();
  while (tok->tag_ == '*' || tok->tag_ == '/' || tok->tag_ == '%') {
    auto rhs = ParseMatmulExpr();
    lhs = BinaryOp::New(tok, lhs, rhs);
    tok = ts_.Next();
  }
  ts_.PutBack();
  return lhs;
}


Expr* Parser::ParseAdditiveExpr() {
  auto lhs = ParseMultiplicativeExpr();
  auto tok = ts_.Next();
  while (tok->tag_ == '+' || tok->tag_ == '-') {
    auto rhs = ParseMultiplicativeExpr();
    lhs = BinaryOp::New(tok, lhs, rhs);

    tok = ts_.Next();
  }

  ts_.PutBack();
  return lhs;
}


Expr* Parser::ParseShiftExpr() {
  auto lhs = ParseAdditiveExpr();
  auto tok = ts_.Next();
  while (tok->tag_ == Token::LEFT || tok->tag_ == Token::RIGHT) {
    auto rhs = ParseAdditiveExpr();
    lhs = BinaryOp::New(tok, lhs, rhs);

    tok = ts_.Next();
  }

  ts_.PutBack();
  return lhs;
}


Expr* Parser::ParseRelationalExpr() {
  auto lhs = ParseShiftExpr();
  auto tok = ts_.Next();
  while (tok->tag_ == Token::LE || tok->tag_ == Token::GE
      || tok->tag_ == '<' || tok->tag_ == '>') {
    auto rhs = ParseShiftExpr();
    lhs = BinaryOp::New(tok, lhs, rhs);

    tok = ts_.Next();
  }

  ts_.PutBack();
  return lhs;
}


Expr* Parser::ParseEqualityExpr() {
  auto lhs = ParseRelationalExpr();
  auto tok = ts_.Next();
  while (tok->tag_ == Token::EQ || tok->tag_ == Token::NE) {
    auto rhs = ParseRelationalExpr();
    lhs = BinaryOp::New(tok, lhs, rhs);

    tok = ts_.Next();
  }

  ts_.PutBack();
  return lhs;
}


Expr* Parser::ParseBitiwiseAndExpr() {
  auto lhs = ParseEqualityExpr();
  auto tok = ts_.Peek();
  while (ts_.Try('&')) {
    auto rhs = ParseEqualityExpr();
    lhs = BinaryOp::New(tok, lhs, rhs);

    tok = ts_.Peek();
  }

  return lhs;
}


Expr* Parser::ParseBitwiseXorExpr() {
  auto lhs = ParseBitiwiseAndExpr();
  auto tok = ts_.Peek();
  while (ts_.Try('^')) {
    auto rhs = ParseBitiwiseAndExpr();
    lhs = BinaryOp::New(tok, lhs, rhs);

    tok = ts_.Peek();
  }

  return lhs;
}


Expr* Parser::ParseBitwiseOrExpr() {
  auto lhs = ParseBitwiseXorExpr();
  auto tok = ts_.Peek();
  while (ts_.Try('|')) {
    auto rhs = ParseBitwiseXorExpr();
    lhs = BinaryOp::New(tok, lhs, rhs);

    tok = ts_.Peek();
  }

  return lhs;
}


Expr* Parser::ParseLogicalAndExpr() {
  auto lhs = ParseBitwiseOrExpr();
  auto tok = ts_.Peek();
  while (ts_.Try(Token::LOGICAL_AND)) {
    auto rhs = ParseBitwiseOrExpr();
    lhs = BinaryOp::New(tok, lhs, rhs);

    tok = ts_.Peek();
  }

  return lhs;
}


Expr* Parser::ParseLogicalOrExpr() {
  auto lhs = ParseLogicalAndExpr();
  auto tok = ts_.Peek();
  while (ts_.Try(Token::LOGICAL_OR)) {
    auto rhs = ParseLogicalAndExpr();
    lhs = BinaryOp::New(tok, lhs, rhs);

    tok = ts_.Peek();
  }

  return lhs;
}


Expr* Parser::ParseConditionalExpr() {
  auto cond = ParseLogicalOrExpr();
  auto tok = ts_.Peek();
  if (ts_.Try('?')) {
    // Non-standard GNU extension
    // a ?: b equals a ? a: c
    auto exprTrue = ts_.Test(':') ? cond: ParseExpr();
    ts_.Expect(':');
    auto exprFalse = ParseConditionalExpr();

    return ConditionalOp::New(tok, cond, exprTrue, exprFalse);
  }
  return cond;
}


Expr* Parser::ParseAssignExpr() {
  // Yes, I know the lhs should be unary expression,
  // let it handled by type checking
  Expr* lhs = ParseConditionalExpr();
  Expr* rhs;

  auto tok = ts_.Next();
  switch (tok->tag_) {
  case Token::MUL_ASSIGN:
    rhs = ParseAssignExpr();
    rhs = BinaryOp::New(tok, '*', lhs, rhs);
    break;

  case Token::DIV_ASSIGN:
    rhs = ParseAssignExpr();
    rhs = BinaryOp::New(tok, '/', lhs, rhs);
    break;

  case Token::MOD_ASSIGN:
    rhs = ParseAssignExpr();
    rhs = BinaryOp::New(tok, '%', lhs, rhs);
    break;

  case Token::ADD_ASSIGN:
    rhs = ParseAssignExpr();
    rhs = BinaryOp::New(tok, '+', lhs, rhs);
    break;

  case Token::SUB_ASSIGN:
    rhs = ParseAssignExpr();
    rhs = BinaryOp::New(tok, '-', lhs, rhs);
    break;

  case Token::LEFT_ASSIGN:
    rhs = ParseAssignExpr();
    rhs = BinaryOp::New(tok, Token::LEFT, lhs, rhs);
    break;

  case Token::RIGHT_ASSIGN:
    rhs = ParseAssignExpr();
    rhs = BinaryOp::New(tok, Token::RIGHT, lhs, rhs);
    break;

  case Token::AND_ASSIGN:
    rhs = ParseAssignExpr();
    rhs = BinaryOp::New(tok, '&', lhs, rhs);
    break;

  case Token::XOR_ASSIGN:
    rhs = ParseAssignExpr();
    rhs = BinaryOp::New(tok, '^', lhs, rhs);
    break;

  case Token::OR_ASSIGN:
    rhs = ParseAssignExpr();
    rhs = BinaryOp::New(tok, '|', lhs, rhs);
    break;

  case '=':
    rhs = ParseAssignExpr();
    break;

  default:
    ts_.PutBack();
    return lhs; // Could be constant
  }

  return BinaryOp::New(tok, '=', lhs, rhs);
}


void Parser::ParseStaticAssert() {
  ts_.Expect('(');
  auto condExpr = ParseAssignExpr();
  ts_.Expect(',');
  auto msg = ConcatLiterals(ts_.Expect(Token::LITERAL));
  ts_.Expect(')');
  ts_.Expect(';');
  if (!Evaluator<long>().Eval(condExpr)) {
    Error(ts_.Peek(), "static assertion failed: %s\n",
          msg->SVal()->c_str());
  }
}


// Return: list of declarations
CompoundStmt* Parser::ParseDecl() {
  StmtList stmts;
  if (ts_.Try(Token::STATIC_ASSERT)) {
    ParseStaticAssert();
  } else {
    int storageSpec, funcSpec, align;
    auto type = ParseDeclSpec(&storageSpec, &funcSpec, &align);
    if (!ts_.Test(';')) {
      do {
        auto ident = ParseDirectDeclarator(type, storageSpec, funcSpec, align);
        auto init = ParseInitDeclarator(ident);
        if (init) stmts.push_back(init);
      } while (ts_.Try(','));
    }
    ts_.Expect(';');
  }

  return CompoundStmt::New(stmts);
}


// For state machine
enum {
  // Compatibility for these key words
  COMP_SIGNED = T_SHORT | T_INT | T_LONG | T_LLONG,
  COMP_UNSIGNED = T_SHORT | T_INT | T_LONG | T_LLONG,
  COMP_CHAR = T_SIGNED | T_UNSIGNED,
  COMP_SHORT = T_SIGNED | T_UNSIGNED | T_INT,
  COMP_INT = T_SIGNED | T_UNSIGNED | T_LONG | T_SHORT | T_LLONG,
  COMP_LONG = T_SIGNED | T_UNSIGNED | T_LONG | T_INT,
  COMP_DOUBLE = T_LONG | T_COMPLEX,
  COMP_COMPLEX = T_FLOAT | T_DOUBLE | T_LONG,

  COMP_THREAD = S_EXTERN | S_STATIC,
};


static inline void TypeLL(int& typeSpec) {
  if (typeSpec & T_LONG) {
    typeSpec &= ~T_LONG;
    typeSpec |= T_LLONG;
  } else {
    typeSpec |= T_LONG;
  }
}


QualType Parser::ParseSpecQual() {
  return ParseDeclSpec(nullptr, nullptr, nullptr);
}


static void EnsureAndSetStorageSpec(const Token* tok, int* storage, int spec) {
  if (!storage)
    Error(tok, "unexpected storage specifier");
  if (*storage != 0)
    Error(tok, "duplicated storage specifier");
  *storage |= spec;
}


/*
 * param: storage: null, only type specifier and qualifier accepted;
 */
QualType Parser::ParseDeclSpec(int* storageSpec, int* funcSpec, int* alignSpec) {
#define ERR_FUNC_SPEC ("unexpected function specifier")
#define ERR_STOR_SPEC ("unexpected storage specifier")
#define ERR_DECL_SPEC ("two or more data types in declaration specifiers")

  QualType type(nullptr);
  int qualSpec = 0;
  int typeSpec = 0;

  if (storageSpec) *storageSpec = 0;
  if (funcSpec) *funcSpec = 0;
  if (alignSpec) *alignSpec = 0;

  const Token* tok;
  for (; ;) {
    tok = ts_.Next();
    switch (tok->tag_) {
    // Function specifier
    case Token::INLINE:
      if (!funcSpec)
        Error(tok, ERR_FUNC_SPEC);
      *funcSpec |= F_INLINE;
      break;

    case Token::NORETURN:
      if (!funcSpec)
        Error(tok, ERR_FUNC_SPEC);
      *funcSpec |= F_NORETURN;
      break;

    // Alignment specifier
    case Token::ALIGNAS: {
      if (!alignSpec)
        Error(tok, "unexpected alignment specifier");
      auto align = ParseAlignas();
      if (align)
        *alignSpec = align;
      break;
    }
    // Storage specifier
    // TODO(wgtdkp): typedef needs more constraints
    case Token::TYPEDEF:
      EnsureAndSetStorageSpec(tok, storageSpec, S_TYPEDEF);
      break;

    case Token::EXTERN:
      EnsureAndSetStorageSpec(tok, storageSpec, S_EXTERN);
      break;

    case Token::GLOBAL:
      EnsureAndSetStorageSpec(tok, storageSpec, S_GLOBAL);
      break;

    case Token::STATIC:
      if (!storageSpec)
        Error(tok, ERR_FUNC_SPEC);
      if (*storageSpec & ~S_THREAD)
        Error(tok, "duplicated storage specifier");
      *storageSpec |= S_STATIC;
      break;

    case Token::THREAD:
      if (!storageSpec)
        Error(tok, ERR_FUNC_SPEC);
      if (*storageSpec & ~COMP_THREAD)
        Error(tok, "duplicated storage specifier");
      *storageSpec |= S_THREAD;
      break;


    // Type qualifier
    case Token::CONST:    qualSpec |= Qualifier::CONST;    break;
    case Token::RESTRICT: qualSpec |= Qualifier::RESTRICT; break;
    case Token::VOLATILE: qualSpec |= Qualifier::VOLATILE; break;
    case Token::CMEM:     qualSpec |= Qualifier::CMEM; break;

    // Type specifier
    case Token::SIGNED:
      if (typeSpec & ~COMP_SIGNED)
        Error(tok, ERR_DECL_SPEC);
      typeSpec |= T_SIGNED;
      break;

    case Token::UNSIGNED:
      if (typeSpec & ~COMP_UNSIGNED)
        Error(tok, ERR_DECL_SPEC);
      typeSpec |= T_UNSIGNED;
      break;

    case Token::VOID:
      if (typeSpec & ~0)
        Error(tok, ERR_DECL_SPEC);
      typeSpec |= T_VOID;
      break;

    case Token::CHAR:
      if (typeSpec & ~COMP_CHAR)
        Error(tok, ERR_DECL_SPEC);
      typeSpec |= T_CHAR;
      break;

    case Token::SHORT:
      if (typeSpec & ~COMP_SHORT)
        Error(tok, ERR_DECL_SPEC);
      typeSpec |= T_SHORT;
      break;

    case Token::INT:
      if (typeSpec & ~COMP_INT)
        Error(tok, ERR_DECL_SPEC);
      typeSpec |= T_INT;
      break;

    case Token::LONG:
      if (typeSpec & ~COMP_LONG)
        Error(tok, ERR_DECL_SPEC);
      TypeLL(typeSpec);
      break;

    case Token::HALF:
      if(typeSpec & ~T_COMPLEX)
        Error(tok, ERR_DECL_SPEC);
      typeSpec |= T_HALF;
      break;

    case Token::FLOAT:
      if (typeSpec & ~T_COMPLEX)
        Error(tok, ERR_DECL_SPEC);
      typeSpec |= T_FLOAT;
      break;

    case Token::DOUBLE:
      if (typeSpec & ~COMP_DOUBLE)
        Error(tok, ERR_DECL_SPEC);
      typeSpec |= T_DOUBLE;
      break;

    case Token::BOOL:
      if (typeSpec != 0)
        Error(tok, ERR_DECL_SPEC);
      typeSpec |= T_BOOL;
      break;

    case Token::COMPLEX:
      if (typeSpec & ~COMP_COMPLEX)
        Error(tok, ERR_DECL_SPEC);
      typeSpec |= T_COMPLEX;
      break;

    case Token::STRUCT:
    case Token::UNION:
      if (typeSpec & ~0)
        Error(tok, ERR_DECL_SPEC);
      type = ParseStructUnionSpec(Token::STRUCT == tok->tag_);
      typeSpec |= T_STRUCT_UNION;
      break;

    case Token::ENUM:
      if (typeSpec != 0)
        Error(tok, ERR_DECL_SPEC);
      type = ParseEnumSpec();
      typeSpec |= T_ENUM;
      break;

    case Token::ATOMIC:
      Error(tok, "atomic not supported");
      break;

    default:
      if (typeSpec == 0 && IsTypeName(tok)) {
        auto ident = curScope_->Find(tok);
        type = ident->Type();
        // We may change the length of a array type by initializer,
        // thus, make a copy of this type.
        auto arrType = type->ToArray();
        if (arrType && !type->Complete())
          type = ArrayType::New(arrType->Len(), arrType->Derived());
        typeSpec |= T_TYPEDEF_NAME;
      } else  {
        goto end_of_loop;
      }
    }
  }

end_of_loop:
  ts_.PutBack();
  switch (typeSpec) {
  case 0:
    Error(tok, "expect type specifier");
    break;

  case T_VOID:
    type = VoidType::New();
    break;

  case T_STRUCT_UNION:
  case T_ENUM:
  case T_TYPEDEF_NAME:
    break;

  default:
    type = ArithmType::New(typeSpec);
    break;
  }
  // GNU extension: type attributes
  //if (storageSpec && (*storageSpec & S_TYPEDEF))
  //  TryAttributeSpecList();

  return QualType(type.GetPtr(), qualSpec | type.Qual());

#undef ERR_FUNC_SPEC
#undef ERR_STOR_SPEC
#undef ERR_DECL_SPEC
}


int Parser::ParseAlignas() {
  int align;
  ts_.Expect('(');
  auto tok = ts_.Peek();
  if (IsTypeName(ts_.Peek())) {
    auto type = ParseTypeName();
    ts_.Expect(')');
    align = type->Align();
  } else {
    auto expr = ParseExpr();
    align = Evaluator<long>().Eval(expr);
    ts_.Expect(')');
  }
  if (align < 0 || ((align - 1) & align))
    Error(tok, "requested alignment is not a positive power of 2");
  return align;
}


Type* Parser::ParseEnumSpec() {
  // GNU extension: type attributes
  TryAttributeSpecList();

  std::string tagName;
  auto tok = ts_.Peek();
  if (ts_.Try(Token::IDENTIFIER)) {
    tagName = tok->str_;
    if (ts_.Try('{')) {
      // 定义enum类型
      auto tagIdent = curScope_->FindTagInCurScope(tok);
      if (!tagIdent) {
        auto type = ArithmType::New(T_INT);
        auto ident = Identifier::New(tok, type, L_NONE);
        curScope_->InsertTag(ident);
        return ParseEnumerator(type);   // 处理反大括号: '}'
      }

      if (!tagIdent->Type()->IsInteger()) // struct/union tag
        Error(tok, "redefinition of enumeration tag '%s'", tagName.c_str());
      return ParseEnumerator(tagIdent->Type()->ToArithm());
    } else {
      auto tagIdent = curScope_->FindTag(tok);
      if (tagIdent) {
        return tagIdent->Type();
      }
      auto type = ArithmType::New(T_INT);
      auto ident = Identifier::New(tok, type, L_NONE);
      curScope_->InsertTag(ident);
      return type;
    }
  }

  ts_.Expect('{');
  auto type = ArithmType::New(T_INT);
  return ParseEnumerator(type);   // 处理反大括号: '}'
}


Type* Parser::ParseEnumerator(ArithmType* type) {
  assert(type && type->IsInteger());
  int val = 0;
  do {
    auto tok = ts_.Expect(Token::IDENTIFIER);
    // GNU extension: enumerator attributes
    TryAttributeSpecList();

    const auto& enumName = tok->str_;
    auto ident = curScope_->FindInCurScope(tok);
    if (ident) {
      Error(tok, "redefinition of enumerator '%s'", enumName.c_str());
    }
    if (ts_.Try('=')) {
      auto expr = ParseAssignExpr();
      val = Evaluator<long>().Eval(expr);
    }
    auto enumer = Enumerator::New(tok, val);
    ++val;
    curScope_->Insert(enumer);
    ts_.Try(',');
  } while (!ts_.Try('}'));

  type->SetComplete(true);
  return type;
}


/*
 * 四种 name space：
 * 1.label, 如 goto end; 它有函数作用域
 * 2.struct/union/enum 的 tag
 * 3.struct/union 的成员
 * 4.其它的普通的变量
 */
Type* Parser::ParseStructUnionSpec(bool isStruct) {
  // GNU extension: type attributes
  TryAttributeSpecList();

  std::string tagName;
  auto tok = ts_.Peek();
  if (ts_.Try(Token::IDENTIFIER)) {
    tagName = tok->str_;
    if (ts_.Try('{')) {
      // 看见大括号，表明现在将定义该struct/union类型
      // 我们不用关心上层scope是否定义了此tag，如果定义了，那么就直接覆盖定义
      auto tagIdent = curScope_->FindTagInCurScope(tok);
      if (!tagIdent) {
        // 现在是在当前scope第一次看到name，所以现在是第一次定义，连前向声明都没有；
        auto type = StructType::New(isStruct, tagName.size(), curScope_);
        auto ident = Identifier::New(tok, type, L_NONE);
        curScope_->InsertTag(ident);
        return ParseStructUnionDecl(type); // 处理反大括号: '}'
      }


      // 在当前scope找到了类型，但可能只是声明；注意声明与定义只能出现在同一个scope；
      // 1.如果声明在定义的外层scope,那么即使在内层scope定义了完整的类型，此声明仍然是无效的；
      //   因为如论如何，编译器都不会在内部scope里面去找定义，所以声明的类型仍然是不完整的；
      // 2.如果声明在定义的内层scope,(也就是先定义，再在内部scope声明)，这时，不完整的声明会覆盖掉完整的定义；
      //   因为编译器总是向上查找符号，不管找到的是完整的还是不完整的，都要；
      if (!tagIdent->Type()->Complete()) {
        // 找到了此tag的前向声明，并更新其符号表，最后设置为complete type
        return ParseStructUnionDecl(tagIdent->Type()->ToStruct());
      } else {
        // 在当前作用域找到了完整的定义，并且现在正在定义同名的类型，所以报错；
        Error(tok, "redefinition of struct tag '%s'", tagName.c_str());
      }
    } else {
      // 没有大括号，表明不是定义一个struct/union;那么现在只可能是在：
      // 1.声明；
      // 2.声明的同时，定义指针(指针允许指向不完整类型) (struct Foo* p; 是合法的) 或者其他合法的类型；
      //   如果现在索引符号表，那么：
      //   1.可能找到name的完整定义，也可能只找得到不完整的声明；不管name指示的是不是完整类型，我们都只能选择name指示的类型；
      //   2.如果我们在符号表里面压根找不到name,那么现在是name的第一次声明，创建不完整的类型并插入符号表；
      auto tagIdent = curScope_->FindTag(tok);

      // 如果tag已经定义或声明，那么直接返回此定义或者声明
      if (tagIdent) {
        return tagIdent->Type();
      }
      // 如果tag尚没有定义或者声明，那么创建此tag的声明(因为没有见到‘{’，所以不会是定义)
      auto type = StructType::New(isStruct, true, curScope_);

      // 因为有tag，所以不是匿名的struct/union， 向当前的scope插入此tag
      auto ident = Identifier::New(tok, type, L_NONE);
      curScope_->InsertTag(ident);
      return type;
    }
  }
  // 没见到identifier，那就必须有struct/union的定义，这叫做匿名struct/union;
  ts_.Expect('{');

  // 现在，如果是有tag，那它没有前向声明；如果是没有tag，那更加没有前向声明；
  // 所以现在是第一次开始定义一个完整的struct/union类型
  auto type = StructType::New(isStruct, tagName.size(), curScope_);
  return ParseStructUnionDecl(type); // 处理反大括号: '}'
}


StructType* Parser::ParseStructUnionDecl(StructType* type) {
#define ADD_MEMBER() {                        \
  auto member = Object::New(tok, memberType); \
  if (align > 0)                              \
    member->SetAlign(align);                  \
  type->AddMember(member);                    \
}

  // 既然是定义，那输入肯定是不完整类型，不然就是重定义了
  assert(type && !type->Complete());

  auto scopeBackup = curScope_;
  curScope_ = type->MemberMap(); // Internal symbol lookup rely on curScope_
  while (!ts_.Try('}')) {
    if (ts_.Empty()) {
      Error(ts_.Peek(), "premature end of input");
    }

    if(ts_.Try(Token::STATIC_ASSERT)) {
      ParseStaticAssert();
      continue;
    }

    // 解析type specifier/qualifier, 不接受storage等
    int align;
    auto baseType = ParseDeclSpec(nullptr, nullptr, &align);
    do {
      auto declInfo = ParseDeclarator(baseType);
      auto tok = declInfo.tok;
      auto memberType = declInfo.type;

      if (ts_.Try(':')) {
        ParseBitField(type, tok, memberType);
        continue;
      }

      if (tok == nullptr) {
        auto suType = memberType->ToStruct();
        if (suType && !suType->HasTag()) {
          auto anony = Object::NewAnony(ts_.Peek(), suType);
          type->MergeAnony(anony);
          continue;
        } else {
          Error(ts_.Peek(), "declaration does not declare anything");
        }
      }

      const auto& name = tok->str_;
      if (type->GetMember(name)) {
        Error(tok, "duplicate member '%s'", name.c_str());
      } else if (!memberType->Complete()) {
        // C11 6.7.2.1 [3]:
        if (type->IsStruct() &&
            // Struct has more than one named member
            type->MemberMap()->size() > 0 &&
            memberType->ToArray()) {
          ts_.Expect(';'); ts_.Expect('}');
          ADD_MEMBER();
          goto finalize;
        } else {
          Error(tok, "field '%s' has incomplete type", name.c_str());
        }
      } else if (memberType->ToFunc()) {
        Error(tok, "field '%s' declared as a function", name.c_str());
      }

      ADD_MEMBER();
    } while (ts_.Try(','));
    ts_.Expect(';');
  }
finalize:
  // GNU extension: type attributes
  TryAttributeSpecList();

  // struct/union定义结束，设置其为完整类型
  type->Finalize();
  type->SetComplete(true);
  // TODO(wgtdkp): we need to export tags defined inside struct
  const auto& tags = curScope_->AllTagsInCurScope();
  for (auto tag: tags) {
    if (scopeBackup->FindTag(tag->Tok()))
      Error(tag, "redefinition of tag '%s'\n", tag->Name().c_str());
    scopeBackup->InsertTag(tag);
  }
  curScope_ = scopeBackup;

  return type;
}


void Parser::ParseBitField(StructType* structType,
                           const Token* tok,
                           QualType type) {
  if (!type->IsInteger()) {
    Error(tok ? tok: ts_.Peek(), "expect integer type for bitfield");
  }

  auto expr = ParseAssignExpr();
  auto width = Evaluator<long>().Eval(expr);
  if (width < 0) {
    Error(expr, "expect non negative value");
  } else if (width == 0 && tok) {
    Error(tok, "no declarator expected for a bitfield with width 0");
  } else if (width > type->Width() * 8) {
    Error(expr, "width exceeds its type");
  }

  auto offset = structType->Offset() - type->Width();
  // C11 6.7.5 [2]: alignment attribute shall not be specified in declaration of a bit field
  // so here is ok to use type->Align()
  offset = Type::MakeAlign(std::max(offset, 0), type->Align());

  int bitFieldOffset;
  unsigned char begin;

  if (!structType->IsStruct()) {
    begin = 0;
    bitFieldOffset = 0;
  } else if (structType->Members().size() == 0) {
    begin = 0;
    bitFieldOffset = 0;
  } else {
    auto last = structType->Members().back();
    auto totalBits = last->Offset() * 8;
    if (last->BitFieldWidth()) {
      totalBits += last->BitFieldEnd();
    } else { // Is not bit field
      totalBits += last->Type()->Width() * 8;
    }

    if (width == 0)
      width = type->Width() * 8 - totalBits; // So posterior bitfield would be packed
    if (width == 0) // A bitfield with zero width is never added to member list
      return;       // Because we use bitfield width to tell if a member is bitfield or not.
    if (width + totalBits <= type->Width() * 8) {
      begin = totalBits % 8;
      bitFieldOffset = totalBits / 8;
    } else {
      begin = 0;
      bitFieldOffset = Type::MakeAlign(structType->Offset(), type->Width());
    }
  }

  Object* bitField;
  if (tok) {
    bitField = Object::New(tok, type, 0, L_NONE, begin, width);
  } else {
    bitField = Object::NewAnony(ts_.Peek(), type, 0, L_NONE, begin, width);
  }
  structType->AddBitField(bitField, bitFieldOffset);
}


int Parser::ParseQual() {
  int qualSpec = 0;
  for (; ;) {
    auto tok = ts_.Next();
    switch (tok->tag_) {
    case Token::CONST:    qualSpec |= Qualifier::CONST;    break;
    case Token::RESTRICT: qualSpec |= Qualifier::RESTRICT; break;
    case Token::VOLATILE: qualSpec |= Qualifier::VOLATILE; break;
    case Token::CMEM:     qualSpec |= Qualifier::CMEM; break;
    case Token::ATOMIC:   Error(tok, "do not support 'atomic'"); break;
    default: ts_.PutBack(); return qualSpec;
    }
  }
}


QualType Parser::ParsePointer(QualType typePointedTo) {
  while (ts_.Try('*')) {
    auto t = PointerType::New(typePointedTo);
    typePointedTo = QualType(t, ParseQual());
  }
  return typePointedTo;
}


static QualType ModifyBase(QualType type, QualType base, QualType newBase) {
  if (type == base)
    return newBase;

  auto ty = type->ToDerived();
  ty->SetDerived(ModifyBase(ty->Derived(), base, newBase));

  return ty;
}


/*
 * Return: pair of token(must be identifier) and it's type
 *     if token is nullptr, then we are parsing abstract declarator
 *     else, parsing direct declarator.
 */
DeclInfo Parser::ParseDeclarator(QualType base) {
  // May be pointer
  auto pointerType = ParsePointer(base);

  if (ts_.Try('(')) {
    // 现在的 pointerType 并不是正确的 base type
    auto declInfo = ParseDeclarator(pointerType);
    auto tok = declInfo.tok;
    auto type = declInfo.type;

    ts_.Expect(')');

    auto newBase = ParseArrayFuncDeclarator(tok, pointerType);

    // 修正 base type
    auto retType = ModifyBase(type, pointerType, newBase);
    return DeclInfo(declInfo.tok, retType);
  } else if (ts_.Peek()->IsIdentifier()) {
    auto tok = ts_.Next();
    // GNU extension: variable attributes
    ASTNode::AttrList attrList = TryAttributeSpecList();
    auto retType = ParseArrayFuncDeclarator(tok, pointerType);
    return DeclInfo(tok, retType, attrList);
  } else {
    errTok_ = ts_.Peek();
    auto retType = ParseArrayFuncDeclarator(nullptr, pointerType);
    return DeclInfo(nullptr, retType);
  }
}


Identifier* Parser::ProcessDeclarator(const Token* tok,
                                      QualType type,
                                      const ASTNode::AttrList& attrs,
                                      int storageSpec,
                                      int funcSpec,
                                      int align) {
  assert(tok);

  // 检查在同一 scope 是否已经定义此变量
  // 如果 storage 是 typedef，那么应该往符号表里面插入 type
  // 定义 void 类型变量是非法的，只能是指向void类型的指针
  // 如果 funcSpec != 0, 那么现在必须是在定义函数，否则出错
  const auto& name = tok->str_;
  Identifier* ident;

  if (storageSpec & S_TYPEDEF) {
    // C11 6.7.5 [2]: alignment specifier
    if (align > 0)
      Error(tok, "alignment specified for typedef");

    ident = curScope_->FindInCurScope(tok);
    if (ident) { // There is prio declaration in the same scope
      // The same declaration, simply return the prio declaration
      if (!type->Compatible(*ident->Type()))
        Error(tok, "conflicting types for '%s'", name.c_str());

      // TODO(wgtdkp): add previous declaration information
      return ident;
    }

    if(!attrs.empty()) {
      Error(tok, "typedef attributes not allowed");
    }

    ident = Identifier::New(tok, type, L_NONE);
    curScope_->Insert(ident);
    return ident;
  }

  if (type->ToVoid()) {
    Error(tok, "variable or field '%s' declared void",
        name.c_str());
  }

  if (type->ToFunc() && curScope_->Type() != S_FILE
      && (storageSpec & S_STATIC)) {
    Error(tok, "invalid storage class for function '%s'", name.c_str());
  }

  Linkage linkage;
  // Identifiers in function prototype have no linkage
  if (curScope_->Type() == S_PROTO) {
    linkage = L_NONE;
  } else if (curScope_->Type() == S_FILE) {
    linkage = L_EXTERNAL; // Default linkage for file scope identifiers
    if (storageSpec & S_STATIC)
      linkage = L_INTERNAL;
  } else if (!(storageSpec & S_EXTERN)) {
    linkage = L_NONE; // Default linkage for block scope identifiers
    if (type->ToFunc())
      linkage = L_EXTERNAL;
  } else {
    linkage = L_EXTERNAL;
  }

  ident = curScope_->FindInCurScope(tok);
  if (ident) { // There is prio declaration in the same scope
    if (!type->Compatible(*ident->Type())) {
      Error(tok, "conflicting types for '%s'", name.c_str());
    }

    // The same scope prio declaration has no linkage,
    // there is a redeclaration error
    if (linkage == L_NONE) {
      Error(tok, "redeclaration of '%s' with no linkage",
          name.c_str());
    } else if (linkage == L_EXTERNAL) {
      if (ident->Linkage() == L_NONE) {
        Error(tok, "conflicting linkage for '%s'", name.c_str());
      }
    } else {
      if (ident->Linkage() != L_INTERNAL) {
        Error(tok, "conflicting linkage for '%s'", name.c_str());
      }
    }
    // The same declaration, simply return the prio declaration
    if (!ident->Type()->Complete())
      ident->Type()->SetComplete(type->Complete());
    // Prio declaration of a function may omit the param name
    if (type->ToFunc())
      ident->Type()->ToFunc()->SetParams(type->ToFunc()->Params());
    else if (ident->ToObject() && !(storageSpec & S_EXTERN))
      ident->ToObject()->SetStorage(ident->ToObject()->Storage() & ~S_EXTERN);
    return ident;
  } else if (linkage == L_EXTERNAL) {
    ident = curScope_->Find(tok);
    if (ident) {
      if (!type->Compatible(*ident->Type())) {
        Error(tok, "conflicting types for '%s'", name.c_str());
      }
      if (ident->Linkage() != L_NONE) {
        linkage = ident->Linkage();
      }
      // Don't return, override it
    } else {
      ident = externalSymbols_->FindInCurScope(tok);
      if (ident) {
        if (!type->Compatible(*ident->Type())) {
          Error(tok, "conflicting types for '%s'", name.c_str());
        }
        // TODO(wgtdkp): ???????
        // Don't return
        // To stop later declaration with the same name in the same scope overriding this declaration

        // Useless here, just keep it
        if (!ident->Type()->Complete())
          ident->Type()->SetComplete(type->Complete());
        //return ident;
      }
    }
  }

  Identifier* ret;
  // TODO(wgtdkp): Treat function as object ?
  if (type->ToFunc()) {
    // C11 6.7.5 [2]: alignment specifier
    if (align > 0)
      Error(tok, "alignment specified for function");
    ret = Identifier::New(tok, type, linkage, attrs);
  } else {
    auto obj = Object::New(tok, type, storageSpec, linkage, 0, 0, attrs);
    if (align > 0)
      obj->SetAlign(align);
    ret = obj;
  }
  curScope_->Insert(ret);
  if (linkage == L_EXTERNAL && ident == nullptr) {
      externalSymbols_->Insert(ret);
  }

  return ret;
}


QualType Parser::ParseArrayFuncDeclarator(const Token* ident, QualType base) {
  if (ts_.Try('[')) {
    if(!base->IsScalar()) {
      Error(ts_.Peek(), "tiles must have scalar elements");
    }
    auto shape = ParseTileShape();
    ts_.Expect(']');
    base = ParseArrayFuncDeclarator(ident, base);
    if (!base->Complete()) {
      Error(ident, "'%s' has incomplete element type", ident->str_.c_str());
    }
    // return a pointer for tiles in constant memory:
    TileType* ret = TileType::New(shape, base);
    if(!ret->CheckPow2NumEl())
      Error(ts_.Peek(), "tile must have power of 2 number of elements");
    return ret;
  } else if (ts_.Try('(')) {	// Function declaration
    if (base->ToFunc()) {
      Error(ts_.Peek(),
          "the return value of function cannot be function");
    } else if (nullptr != base->ToArray()) {
      Error(ts_.Peek(),
          "the return value of function cannot be array");
    }

    FuncType::ParamList params;
    EnterProto();
    auto variadic = ParseParamList(params);
    ExitProto();

    ts_.Expect(')');
    base = ParseArrayFuncDeclarator(ident, base);

    return FuncType::New(base, 0, variadic, params);
  }


  return base;
}


/*
 * Return: -1, length not specified
 */
int Parser::ParseArrayLength() {
  auto hasStatic = ts_.Try(Token::STATIC);
  auto qual = ParseQual();
  if (0 != qual)
    hasStatic = ts_.Try(Token::STATIC);

  // 不支持变长数组
  if (!hasStatic && ts_.Test(']'))
    return -1;

  auto expr = ParseAssignExpr();
  EnsureInteger(expr);
  auto ret = Evaluator<long>().Eval(expr);
  if (ret < 0) {
    Error(expr, "size of array is negative");
  }
  return ret;
}

TileType::ShapeInt Parser::ParseTileShape() {
  TileType::ShapeInt ret;
  size_t i = 0;
  do {
    Expr* expr = ParseConditionalExpr();
    EnsureInteger(expr);
    int dim = Evaluator<long>().Eval(expr);
    if (dim < 0)
      Error(expr, "shape %d of tile is negative", i);
    ret.push_back(dim);
    i++;
  }while(ts_.Try(','));
  return ret;
}

/*
 * Return: true, variadic;
 */
bool Parser::ParseParamList(FuncType::ParamList& params) {
  if (ts_.Test(')'))
    return false;
  auto param = ParseParamDecl();
  if (param->Type()->ToVoid())
    return false;
  params.push_back(param);

  while (ts_.Try(',')) {
    if (ts_.Try(Token::ELLIPSIS))
      return true;
    param = ParseParamDecl();
    if (param->Type()->ToVoid())
      Error(param, "'void' must be the only parameter");
    params.push_back(param);
  }
  return false;
}


Object* Parser::ParseParamDecl() {
  int storageSpec, funcSpec;
  // C11 6.7.5 [2]: alignment specifier cannot be specified in params
  auto type = ParseDeclSpec(&storageSpec, &funcSpec, nullptr);
  auto tokTypePair = ParseDeclarator(type);
  auto tok = tokTypePair.tok;
  QualType fullType(tokTypePair.type.GetPtr(), type.Qual());
  type = Type::MayCast(fullType, true);
  auto attrs = tokTypePair.attrs;
  if (!tok) { // Abstract declarator
    return Object::NewAnony(ts_.Peek(), type, 0, Linkage::L_NONE);
  }

  // Align set to non positive, stands for not specified
  auto ident = ProcessDeclarator(tok, type, attrs, storageSpec, funcSpec, -1);
  if (!ident->ToObject())
    Error(ident, "expect object in param list");

  return ident->ToObject();
}


QualType Parser::ParseAbstractDeclarator(QualType type) {
  auto declInfo = ParseDeclarator(type);
  auto tok = declInfo.tok;
  type = declInfo.type;
  if (tok) { // Not a abstract declarator!
    Error(tok, "unexpected identifier '%s'", tok->str_.c_str());
  }
  return type;
}


Identifier* Parser::ParseDirectDeclarator(QualType type,
                                          int storageSpec,
                                          int funcSpec,
                                          int align) {
  auto declInfo = ParseDeclarator(type);
  auto tok = declInfo.tok;
  type = declInfo.type;
  auto attrs = declInfo.attrs;
  if (tok == nullptr) {
    Error(errTok_, "expect identifier or '('");
  }

  return ProcessDeclarator(tok, type, attrs, storageSpec, funcSpec, align);
}


Declaration* Parser::ParseInitDeclarator(Identifier* ident) {
  auto obj = ident->ToObject();
  if (!obj) { // Do not record function Declaration
    return nullptr;
  }

  const auto& name = obj->Name();
  if (ts_.Try('=')) {
    return ParseInitDeclaratorSub(obj);
  }

  if (!obj->Type()->Complete()) {
    if (obj->Linkage() == L_NONE) {
      Error(obj, "storage size of '%s' isn’t known", name.c_str());
    }
    // FIXME(wgtdkp):
    // Discards the incomplete object declarations
    // It causes linking failure of forward-declared objects with imcomplete type
    return nullptr;
  }

  if (!obj->Decl()) {
    auto decl = Declaration::New(obj);
    obj->SetDecl(decl);
    return decl;
  }

  return nullptr;
}


Declaration* Parser::ParseInitDeclaratorSub(Object* obj) {
  const auto& name = obj->Name();
  if ((curScope_->Type() != S_FILE) && obj->Linkage() != L_NONE) {
    Error(obj, "'%s' has both 'extern' and initializer", name.c_str());
  }

  if (!obj->Type()->Complete() && !obj->Type()->ToArray()) {
    Error(obj, "variable '%s' has initializer but incomplete type",
        name.c_str());
  }

  if (obj->HasInit()) {
    Error(obj, "redefinition of variable '%s'", name.c_str());
  }

  // There could be more than one declaration for
  // an object in the same scope.
  // But it must has external or internal linkage.
  // So, for external/internal objects,
  // the initialization will always go to
  // the first declaration. As the initialization
  // is evaluated at compile time,
  // the order doesn't matter.
  // For objects with no linkage, there is
  // always only one declaration.
  // Once again, we need not to worry about
  // the order of the initialization.
  if (obj->Decl()) {
    ParseInitializer(obj->Decl(), obj->Type(), 0, false, true);
    return nullptr;
  } else {
    auto decl = Declaration::New(obj);
    ParseInitializer(decl, obj->Type(), 0, false, true);
    obj->SetDecl(decl);
    return decl;
  }
}


void Parser::ParseInitializer(Declaration* decl,
                              QualType type,
                              int offset,
                              bool designated,
                              bool forceBrace,
                              unsigned char bitFieldBegin,
                              unsigned char bitFieldWidth) {
  if (designated && !ts_.Test('.') && !ts_.Test('[')) {
    ts_.Expect('=');
  }

//  std::cout << "parsing initialized " << decl->Obj()->Name() << std::endl;
  Expr* expr;
  auto arrType = type->ToArray();
  auto structType = type->ToStruct();
  // A compound literal in initializer is reduced to a initializer directly
  // It means that the compound literal will never be created
  //auto literalType = TryCompoundLiteral();
  //if (literalType && !literalType->Compatible(*type))
  //    Error("incompatible type of initializer");
  if (arrType) {
    if (forceBrace && !ts_.Test('{') && !ts_.Test(Token::LITERAL)) {
      ts_.Expect('{');
    } else if (!ParseLiteralInitializer(decl, arrType, offset)) {
      ParseArrayInitializer(decl, arrType, offset, designated);
      arrType->SetComplete(true);
    }
    return;
  } else if (structType) {
    if (!ts_.Test('.') && !ts_.Test('{')) {
      auto mark = ts_.Mark();
      expr = ParseAssignExpr();
      if (structType->Compatible(*expr->Type())) {
        decl->AddInit({structType, offset, expr});
        return;
      }
      ts_.ResetTo(mark);
      if (forceBrace)
        ts_.Expect('{');
    }
    return ParseStructInitializer(decl, structType, offset, designated);
  }

  // Scalar type
  auto hasBrace = ts_.Try('{');
  expr = ParseAssignExpr();
  if (hasBrace) {
    ts_.Try(',');
    ts_.Expect('}');
  }
  decl->AddInit({type.GetPtr(), offset, expr, bitFieldBegin, bitFieldWidth});
}


bool Parser::ParseLiteralInitializer(Declaration* decl,
                                     ArrayType* type,
                                     int offset) {
  if (!type->Derived()->IsInteger())
    return false;

  auto hasBrace = ts_.Try('{');
  if (!ts_.Test(Token::LITERAL)) {
    if (hasBrace) ts_.PutBack();
    return false;
  }
  auto literal = ConcatLiterals(ts_.Next());
  auto tok = literal->Tok();

  if (hasBrace) {
    ts_.Try(',');
    ts_.Expect('}');
  }

  if (!type->Complete()) {
    type->SetLen(literal->Type()->ToArray()->Len());
    type->SetComplete(true);
  }

  auto width = std::min(type->Width(), literal->Type()->Width());
  auto str = literal->SVal()->c_str();

  for (; width >= 8; width -= 8) {
    auto p = reinterpret_cast<const long*>(str);
    auto type = ArithmType::New(T_LONG);
    auto val = Constant::New(tok, T_LONG, static_cast<long>(*p));
    decl->AddInit({type, offset, val});
    offset += 8;
    str += 8;
  }

  for (; width >= 4; width -= 4) {
    auto p = reinterpret_cast<const int*>(str);
    auto type = ArithmType::New(T_INT);
    auto val = Constant::New(tok, T_INT, static_cast<long>(*p));
    decl->AddInit({type, offset, val});
    offset += 4;
    str += 4;
  }

  for (; width >= 2; width -= 2) {
    auto p = reinterpret_cast<const short*>(str);
    auto type = ArithmType::New(T_SHORT);
    auto val = Constant::New(tok, T_SHORT, static_cast<long>(*p));
    decl->AddInit({type, offset, val});
    offset += 2;
    str += 2;
  }

  for (; width >= 1; --width) {
    auto p = str;
    auto type = ArithmType::New(T_CHAR);
    auto val = Constant::New(tok, T_CHAR, static_cast<long>(*p));
    decl->AddInit({type, offset, val});
    offset++;
    str++;
  }

  return true;
}


void Parser::ParseArrayInitializer(Declaration* decl,
                                   ArrayType* type,
                                   int offset,
                                   bool designated) {
  assert(type);

  if (!type->Complete())
    type->SetLen(0);

  int idx = 0;
  auto width = type->Derived()->Width();
  auto hasBrace = ts_.Try('{');
  while (true) {
    if (ts_.Test('}')) {
      if (hasBrace)
        ts_.Next();
      return;
    }

    if (!designated && !hasBrace && (ts_.Test('.') || ts_.Test('['))) {
      ts_.PutBack(); // Put the read comma(',') back
      return;
    } else if ((designated = ts_.Try('['))) {
      auto expr = ParseAssignExpr();
      EnsureInteger(expr);
      idx = Evaluator<long>().Eval(expr);
      ts_.Expect(']');

      if (idx < 0 || (type->Complete() && idx >= type->Len())) {
        Error(ts_.Peek(), "excess elements in array initializer");
      }
    }

    ParseInitializer(decl, type->Derived(), offset + idx * width, designated);
    designated = false;
    ++idx;

    if (type->Complete() && idx >= type->Len()) {
      break;
    } else if (!type->Complete()) {
      type->SetLen(std::max(idx, type->Len()));
    }

    // Needless comma at the end is legal
    if (!ts_.Try(',')) {
      if (hasBrace)
        ts_.Expect('}');
      return;
    }
  }

  if (hasBrace) {
    ts_.Try(',');
    if (!ts_.Try('}')) {
      Error(ts_.Peek(), "excess elements in array initializer");
    }
  }
}


StructType::Iterator Parser::ParseStructDesignator(StructType* type,
                                                   const std::string& name) {
  auto iter = type->Members().begin();
  for (; iter != type->Members().end(); ++iter) {
    if ((*iter)->Anonymous()) {
      auto anonyType = (*iter)->Type()->ToStruct();
      assert(anonyType);
      if (anonyType->GetMember(name)) {
        return iter; // ParseStructDesignator(anonyType);
      }
    } else if ((*iter)->Name() == name) {
      return iter;
    }
  }
  assert(false);
  return iter;
}


void Parser::ParseStructInitializer(Declaration* decl,
                                    StructType* type,
                                    int offset,
                                    bool designated) {
  assert(type);

  auto hasBrace = ts_.Try('{');
  auto member = type->Members().begin();
  while (true) {
    if (ts_.Test('}')) {
      if (hasBrace)
        ts_.Next();
      return;
    }

    if (!designated && !hasBrace && (ts_.Test('.') || ts_.Test('['))) {
      ts_.PutBack(); // Put the read comma(',') back
      return;
    }

    if ((designated = ts_.Try('.'))) {
      auto tok = ts_.Expect(Token::IDENTIFIER);
      const auto& name = tok->str_;
      if (!type->GetMember(name)) {
        Error(tok, "member '%s' not found", name.c_str());
      }
      member = ParseStructDesignator(type, name);
    }
    if (member == type->Members().end())
      break;

    if ((*member)->Anonymous()) {
      if (designated) { // Put back '.' and member name.
        ts_.PutBack();
        ts_.PutBack();
      }
      // Because offsets of member of anonymous struct/union are based
      // directly on external struct/union
      ParseInitializer(decl, (*member)->Type(), offset, designated, false,
                       (*member)->BitFieldBegin(), (*member)->BitFieldWidth());
    } else {
      ParseInitializer(decl, (*member)->Type(),
                       offset + (*member)->Offset(), designated, false,
                       (*member)->BitFieldBegin(), (*member)->BitFieldWidth());
    }
    designated = false;
    ++member;

    // Union, just init the first member
    if (!type->IsStruct())
      break;

    if (!hasBrace && member == type->Members().end())
      break;

    // Needless comma at the end is allowed
    if (!ts_.Try(',')) {
      if (hasBrace)
        ts_.Expect('}');
      return;
    }
  }

  if (hasBrace) {
    ts_.Try(',');
    if (!ts_.Try('}')) {
      Error(ts_.Peek(), "excess members in struct initializer");
    }
  }
}


/*
 * Statements
 */

Stmt* Parser::ParseStmt() {
  auto tok = ts_.Next();
  if (tok->IsEOF())
    Error(tok, "premature end of input");

  switch (tok->tag_) {
  // GNU extension: statement attributes
  case Token::ATTRIBUTE:
    TryAttributeSpecList();
  case ';':
    return EmptyStmt::New();
  case '{':
    return ParseCompoundStmt();
  case Token::IF:
    return ParseIfStmt();
  case Token::SWITCH:
    return ParseSwitchStmt();
  case Token::WHILE:
    return ParseWhileStmt();
  case Token::DO:
    return ParseDoStmt();
  case Token::FOR:
    return ParseForStmt();
  case Token::GOTO:
    return ParseGotoStmt();
  case Token::CONTINUE:
    return ParseContinueStmt();
  case Token::BREAK:
    return ParseBreakStmt();
  case Token::RETURN:
    return ParseReturnStmt();
  case Token::CASE:
    return ParseCaseStmt();
  case Token::DEFAULT:
    return ParseDefaultStmt();
  }

  if (tok->IsIdentifier() && ts_.Try(':')) {
    // GNU extension: label attributes
    TryAttributeSpecList();
    return ParseLabelStmt(tok);
  }

  ts_.PutBack();
  auto expr = ParseExpr();
  ts_.Expect(';');

  return expr;
}


CompoundStmt* Parser::ParseCompoundStmt(FuncType* funcType) {
  EnterBlock(funcType);

  std::list<Stmt*> stmts;

  while (!ts_.Try('}')) {
    if (ts_.Peek()->IsEOF()) {
      Error(ts_.Peek(), "premature end of input");
    }

    if (IsType(ts_.Peek())) {
      stmts.push_back(ParseDecl());
    } else {
      stmts.push_back(ParseStmt());
    }
  }

  auto scope = curScope_;
  ExitBlock();

  return CompoundStmt::New(stmts, scope);
}


IfStmt* Parser::ParseIfStmt() {
  ts_.Expect('(');
  auto tok = ts_.Peek();
  auto cond = ParseExpr();
  if (!cond->Type()->IsScalar()) {
    Error(tok, "expect scalar");
  }
  ts_.Expect(')');

  auto then = ParseStmt();
  Stmt* els = nullptr;
  if (ts_.Try(Token::ELSE))
    els = ParseStmt();

  return IfStmt::New(cond, then, els);
}


/*
 * for 循环结构：
 *      for (declaration; expression1; expression2) statement
 * 展开后的结构：
 *		declaration
 * cond: if (expression1) then empty
 *		else goto end
 *		statement
 * step: expression2
 *		goto cond
 * next:
 */

#define ENTER_LOOP_BODY(breakDest, continueDest)  \
{                                                 \
  LabelStmt* breakDestBackup = breakDest_;        \
  LabelStmt* continueDestBackup = continueDest_;  \
  breakDest_ = breakDest;                         \
  continueDest_ = continueDest;

#define EXIT_LOOP_BODY()              \
  breakDest_ = breakDestBackup;       \
  continueDest_ = continueDestBackup; \
}

ForStmt* Parser::ParseForStmt() {
  EnterBlock();
  ts_.Expect('(');
  // init
  Stmt* init = nullptr;
  if (IsType(ts_.Peek())) {
    init = ParseDecl();
  } else if (!ts_.Try(';')) {
    init = ParseExpr();
    ts_.Expect(';');
  }
  // cond
  Expr* cond = nullptr;
  if (!ts_.Try(';')) {
    cond = ParseExpr();
    ts_.Expect(';');
  }
  // step
  Expr* step = nullptr;
  if (!ts_.Try(')')) {
    step = ParseExpr();
    ts_.Expect(')');
  }
  // body
  Stmt* body = ParseStmt();
  ExitBlock();
  return ForStmt::New(body, init, cond, step);
}


/*
 * while 循环结构：
 * while (expression) statement
 * 展开后的结构：
 * cond: if (expression) then empty
 *		else goto end
 *		statement
 *		goto cond
 * end:
 */
CompoundStmt* Parser::ParseWhileStmt() {
  std::list<Stmt*> stmts;
  ts_.Expect('(');
  auto tok = ts_.Peek();
  auto condExpr = ParseExpr();
  ts_.Expect(')');

  if (!condExpr->Type()->IsScalar()) {
    Error(tok, "scalar expression expected");
  }

  auto condLabel = LabelStmt::New();
  auto endLabel = LabelStmt::New();
  auto gotoEndStmt = JumpStmt::New(endLabel);
  auto ifStmt = IfStmt::New(condExpr, EmptyStmt::New(), gotoEndStmt);
  stmts.push_back(condLabel);
  stmts.push_back(ifStmt);

  Stmt* bodyStmt;
  ENTER_LOOP_BODY(endLabel, condLabel)
  bodyStmt = ParseStmt();
  EXIT_LOOP_BODY()

  stmts.push_back(bodyStmt);
  stmts.push_back(JumpStmt::New(condLabel));
  stmts.push_back(endLabel);

  return CompoundStmt::New(stmts);
}


/*
 * do-while 循环结构：
 *      do statement while (expression)
 * 展开后的结构：
 * begin: statement
 * cond: if (expression) then goto begin
 *		 else goto end
 * end:
 */
CompoundStmt* Parser::ParseDoStmt() {
  auto beginLabel = LabelStmt::New();
  auto condLabel = LabelStmt::New();
  auto endLabel = LabelStmt::New();

  Stmt* bodyStmt;
  ENTER_LOOP_BODY(endLabel, beginLabel)
  bodyStmt = ParseStmt();
  EXIT_LOOP_BODY()

  ts_.Expect(Token::WHILE);
  ts_.Expect('(');
  auto condExpr = ParseExpr();
  ts_.Expect(')');
  ts_.Expect(';');

  auto gotoBeginStmt = JumpStmt::New(beginLabel);
  auto gotoEndStmt = JumpStmt::New(endLabel);
  auto ifStmt = IfStmt::New(condExpr, gotoBeginStmt, gotoEndStmt);

  std::list<Stmt*> stmts;
  stmts.push_back(beginLabel);
  stmts.push_back(bodyStmt);
  stmts.push_back(condLabel);
  stmts.push_back(ifStmt);
  stmts.push_back(endLabel);

  return CompoundStmt::New(stmts);
}


#undef ENTER_LOOP_BODY
#undef EXIT_LOOP_BODY


#define ENTER_SWITCH_BODY(breakDest, caseLabels)  \
{                                                 \
  CaseLabelList* caseLabelsBackup = caseLabels_;  \
  LabelStmt* defaultLabelBackup = defaultLabel_;  \
  LabelStmt* breakDestBackup = breakDest_;        \
  breakDest_ = breakDest;                         \
  caseLabels_ = &caseLabels;                      \
  defaultLabel_ = nullptr;

#define EXIT_SWITCH_BODY()            \
  caseLabels_ = caseLabelsBackup;     \
  breakDest_ = breakDestBackup;       \
  defaultLabel_ = defaultLabelBackup; \
}


/*
 * switch
 *  jump stmt (skip case labels)
 *  case labels
 *  jump stmts
 *  default jump stmt
 */
CompoundStmt* Parser::ParseSwitchStmt() {
  std::list<Stmt*> stmts;
  ts_.Expect('(');
  auto tok = ts_.Peek();
  auto expr = ParseExpr();
  ts_.Expect(')');

  if (!expr->Type()->IsInteger()) {
    Error(tok, "switch quantity not an integer");
  }

  auto testLabel = LabelStmt::New();
  auto endLabel = LabelStmt::New();
  auto t = TempVar::New(expr->Type());
  auto assign = BinaryOp::New(tok, '=', t, expr);
  stmts.push_back(assign);
  stmts.push_back(JumpStmt::New(testLabel));

  CaseLabelList caseLabels;
  ENTER_SWITCH_BODY(endLabel, caseLabels);

  auto bodyStmt = ParseStmt(); // Fill caseLabels and defaultLabel
  stmts.push_back(bodyStmt);
  stmts.push_back(JumpStmt::New(endLabel));
  stmts.push_back(testLabel);

  for (auto iter = caseLabels.begin();
       iter != caseLabels.end(); ++iter) {
    auto cond = BinaryOp::New(tok, Token::EQ, t, iter->first);
    auto then = JumpStmt::New(iter->second);
    auto ifStmt = IfStmt::New(cond, then, nullptr);
    stmts.push_back(ifStmt);
  }
  if (defaultLabel_)
    stmts.push_back(JumpStmt::New(defaultLabel_));
  EXIT_SWITCH_BODY();

  stmts.push_back(endLabel);

  return CompoundStmt::New(stmts);
}


#undef ENTER_SWITCH_BODY
#undef EXIT_SWITCH_BODY


CompoundStmt* Parser::ParseCaseStmt() {
  auto tok = ts_.Peek();

  // Case ranges: Non-standard GNU extension
  long begin, end;
  begin = Evaluator<long>().Eval(ParseAssignExpr());
  if (ts_.Try(Token::ELLIPSIS))
    end = Evaluator<long>().Eval(ParseAssignExpr());
  else
    end = begin;
  ts_.Expect(':');

  auto labelStmt = LabelStmt::New();
  for (auto val = begin; val <= end; ++val) {
    if (val > INT_MAX)
      Error(tok, "case range exceed range of int");
    auto cons = Constant::New(tok, T_INT, val);
    caseLabels_->push_back(std::make_pair(cons, labelStmt));
  }

  std::list<Stmt*> stmts;
  stmts.push_back(labelStmt);
  stmts.push_back(ParseStmt());

  return CompoundStmt::New(stmts);
}


CompoundStmt* Parser::ParseDefaultStmt() {
  auto tok = ts_.Peek();
  ts_.Expect(':');
  if (defaultLabel_) { // There is a 'default' stmt
    Error(tok, "multiple default labels in one switch");
  }
  auto labelStmt = LabelStmt::New();
  defaultLabel_ = labelStmt;

  std::list<Stmt*> stmts;
  stmts.push_back(labelStmt);
  stmts.push_back(ParseStmt());

  return CompoundStmt::New(stmts);
}


JumpStmt* Parser::ParseContinueStmt() {
  auto tok = ts_.Peek();
  ts_.Expect(';');
  if (continueDest_ == nullptr) {
    Error(tok, "'continue' is allowed only in loop");
  }

  return JumpStmt::New(continueDest_);
}


JumpStmt* Parser::ParseBreakStmt() {
  auto tok = ts_.Peek();
  ts_.Expect(';');
  if (breakDest_ == nullptr) {
    Error(tok, "'break' is allowed only in switch/loop");
  }

  return JumpStmt::New(breakDest_);
}


ReturnStmt* Parser::ParseReturnStmt() {
  Expr* expr;

  if (ts_.Try(';')) {
    expr = nullptr;
  } else {
    expr = ParseExpr();
    ts_.Expect(';');

    auto retType = curFunc_->FuncType()->Derived();
    expr = Expr::MayCast(expr, retType);
  }

  return ReturnStmt::New(expr);
}


JumpStmt* Parser::ParseGotoStmt() {
  auto label = ts_.Peek();
  ts_.Expect(Token::IDENTIFIER);
  ts_.Expect(';');

  auto labelStmt = FindLabel(label->str_);
  if (labelStmt) {
    return JumpStmt::New(labelStmt);
  }

  auto unresolvedJump = JumpStmt::New(nullptr);
  unresolvedJumps_.push_back(std::make_pair(label, unresolvedJump));

  return unresolvedJump;
}


CompoundStmt* Parser::ParseLabelStmt(const Token* label) {
  const auto& labelStr = label->str_;
  auto stmt = ParseStmt();
  if (nullptr != FindLabel(labelStr)) {
    Error(label, "redefinition of label '%s'", labelStr.c_str());
  }

  auto labelStmt = LabelStmt::New();
  AddLabel(labelStr, labelStmt);
  std::list<Stmt*> stmts;
  stmts.push_back(labelStmt);
  stmts.push_back(stmt);

  return CompoundStmt::New(stmts);
}


bool Parser::IsBuiltin(const std::string& name) {
  return name == "__builtin_va_arg" ||
         name == "__builtin_va_start";
}


bool Parser::IsBuiltin(FuncType* type) {
  assert(vaStartType_ && vaArgType_);
  return type == vaStartType_ || type == vaArgType_;
}


// Builtin functions will be inlined
void Parser::DefineBuiltins() {
  // FIXME: potential bug: using same object for params!!!
  auto voidPtr = PointerType::New(VoidType::New());
  auto param = Object::New(nullptr, voidPtr);
  FuncType::ParamList pl;
  pl.push_back(param);
  pl.push_back(param);
  vaStartType_ = FuncType::New(VoidType::New(), F_INLINE, false, pl);
  vaArgType_ = FuncType::New(voidPtr, F_INLINE, false, pl);
}


Identifier* Parser::GetBuiltin(const Token* tok) {
  assert(vaStartType_ && vaArgType_);
  static Identifier* vaStart = nullptr;
  static Identifier* vaArg = nullptr;
  const auto& name = tok->str_;
  if (name == "__builtin_va_start") {
    if (!vaStart)
      vaStart = Identifier::New(tok, vaStartType_, Linkage::L_EXTERNAL);
    return vaStart;
  } else if (name == "__builtin_va_arg") {
    if (!vaArg)
      vaArg = Identifier::New(tok, vaArgType_, Linkage::L_EXTERNAL);
    return vaArg;
  }
  assert(false);
  return nullptr;
}


/*
 * GNU extensions
 */

// Attribute
ASTNode::AttrList Parser::TryAttributeSpecList() {
  ASTNode::AttrList attrList;
  while (ts_.Try(Token::ATTRIBUTE))
    ParseAttributeSpec(attrList);
  return attrList;
}


void Parser::ParseAttributeSpec(ASTNode::AttrList& attrList) {
  ts_.Expect('(');
  ts_.Expect('(');

  while (!ts_.Try(')')) {
    attrList.push_back(ParseAttribute());
    if (!ts_.Try(',')) {
      ts_.Expect(')');
      break;
    }
  }
  ts_.Expect(')');
}


ASTNode::Attr Parser::ParseAttribute() {
  ASTNode::Attr ret;
  if (!ts_.Test(Token::IDENTIFIER))
    return ret;
  auto tok = ts_.Next();
  std::string name = tok->str_;
  // set kind
  if(name == "aligned")
    ret.kind = ASTNode::Attr::ALIGNED;
  else if(name == "readonly")
    ret.kind = ASTNode::Attr::READONLY;
  else if(name == "writeonly")
    ret.kind = ASTNode::Attr::WRITEONLY;
  else if(name == "multipleof")
    ret.kind = ASTNode::Attr::MULTIPLEOF;
  else if(name == "noalias")
    ret.kind = ASTNode::Attr::NOALIAS;
  else if(name == "retune")
    ret.kind = ASTNode::Attr::RETUNE;
  else
    Error(tok, "unknown attribute kind");
  // set exprs
  if (ts_.Try('(')) {
    if (ts_.Try(')'))
      return ret;
    ret.vals.push_back(ParseExpr());
    if (ts_.Test(',')) {
      while (ts_.Try(',')) {}
    }
    ts_.Try(')');
  }
  return ret;
}
