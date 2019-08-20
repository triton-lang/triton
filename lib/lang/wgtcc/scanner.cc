#include "triton/lang/wgtcc/scanner.h"

#include <cctype>
#include <climits>


void Scanner::Tokenize(TokenSequence& ts) {
  while (true) {
    auto tok = Scan();
    if (tok->tag_ == Token::END) {
      if (ts.Empty() || (ts.Back()->tag_ != Token::NEW_LINE)) {
        auto t = Token::New(*tok);
        t->tag_ = Token::NEW_LINE;
        t->str_ = "\n";
        ts.InsertBack(t);
      }
      break;
    } else {
      if (!ts.Empty() && ts.Back()->tag_ == Token::NEW_LINE)
        tok->ws_ = true;
      ts.InsertBack(tok);
    }
  }
}


std::string Scanner::ScanHeadName(const Token* lhs, const Token* rhs) {
  std::string str;
  const char* begin = lhs->loc_.Begin() + 1;
  const char* end = rhs->loc_.Begin();
  for (; begin != end; ++begin) {
    if (*begin == '\n' && str.back() == '\\')
      str.pop_back();
    else
      str.push_back(*begin);
  }
  return str;
}


Token* Scanner::Scan(bool ws) {
  tok_.ws_ = ws;
  SkipWhiteSpace();

  Mark();

  if (Test('\n')) {
    auto ret = MakeNewLine();
    Next();
    return ret;
  }
  auto c = Next();
  switch (c) {
  case '#': return MakeToken(Try('#') ? Token::DSHARP: c);
  case ':': return MakeToken(Try('>') ? ']': c);
  case '(': case ')': case '[': case ']':
  case '?': case ',': case '{': case '}':
  case '~': case ';': case '@':
    return MakeToken(c);
  case '-':
    if (Try('>')) return MakeToken(Token::PTR);
    if (Try('-')) return MakeToken(Token::DEC);
    if (Try('=')) return MakeToken(Token::SUB_ASSIGN);
    return MakeToken(c);
  case '+':
    if (Try('+')) return MakeToken(Token::INC);
    if (Try('=')) return MakeToken(Token::ADD_ASSIGN);
    return MakeToken(c);
  case '<':
    if (Try('<')) return MakeToken(Try('=') ? Token::LEFT_ASSIGN: Token::LEFT);
    if (Try('=')) return MakeToken(Token::LE);
    if (Try(':')) return MakeToken('[');
    if (Try('%')) return MakeToken('{');
    return MakeToken(c);
  case '%':
    if (Try('=')) return MakeToken(Token::MOD_ASSIGN);
    if (Try('>')) return MakeToken('}');
    if (Try(':')) {
      if (Try('%')) {
        if (Try(':')) return MakeToken(Token::DSHARP);
        PutBack();
      }
      return MakeToken('#');
    }
    return MakeToken(c);
  case '>':
    if (Try('>')) return MakeToken(Try('=') ? Token::RIGHT_ASSIGN: Token::RIGHT);
    if (Try('=')) return MakeToken(Token::GE);
    return MakeToken(c);
  case '=': return MakeToken(Try('=') ? Token::EQ: c);
  case '!': return MakeToken(Try('=') ? Token::NE: c);
  case '&':
    if (Try('&')) return MakeToken(Token::LOGICAL_AND);
    if (Try('=')) return MakeToken(Token::AND_ASSIGN);
    return MakeToken(c);
  case '|':
    if (Try('|')) return MakeToken(Token::LOGICAL_OR);
    if (Try('=')) return MakeToken(Token::OR_ASSIGN);
    return MakeToken(c);
  case '*': return MakeToken(Try('=') ? Token::MUL_ASSIGN: c);
  case '/':
    if (Test('/') || Test('*')) {
      SkipComment();
      return Scan(true);
    }
    return MakeToken(Try('=') ? Token::DIV_ASSIGN: c);
  case '^': return MakeToken(Try('=') ? Token::XOR_ASSIGN: c);
  case '.':
    if (isdigit(Peek())) return SkipNumber();
    if (Try('.')) {
      if (Try('.')) return MakeToken(Token::ELLIPSIS);
      PutBack();
      return MakeToken('.');
    }
    return MakeToken(c);
  case '0' ... '9': return SkipNumber();
  case 'u': case 'U': case 'L': {
    /*auto enc = */ScanEncoding(c);
    if (Try('\'')) return SkipCharacter();
    if (Try('\"')) return SkipLiteral();
    return SkipIdentifier();
  }
  case '\'': return SkipCharacter();
  case '\"': return SkipLiteral();
  case 'a' ... 't': case 'v' ... 'z': case 'A' ... 'K':
  case 'M' ... 'T': case 'V' ... 'Z': case '_': case '$':
  case 0x80 ... 0xfd:
    return SkipIdentifier();
  case '\\':
    // Universal character name is allowed in identifier
    if (Test('u') || Test('U'))
      return SkipIdentifier();
    return MakeToken(Token::INVALID);
  case '\0': return MakeToken(Token::END);
  default: return MakeToken(Token::INVALID);
  }
}


void Scanner::SkipWhiteSpace() {
  while (isspace(Peek()) && Peek() != '\n') {
    tok_.ws_ = true;
    Next();
  }
}


void Scanner::SkipComment() {
  if (Try('/')) {
    // Line comment terminated an newline or eof
    while (!Empty()) {
      if (Peek() == '\n')
        return;
      Next();
    }
    return;
  } else if (Try('*')) {
    while (!Empty()) {
      auto c = Next();
      if (c  == '*' && Peek() == '/') {
        Next();
        return;
      }
    }
    Error(loc_, "unterminated block comment");
  }
  assert(false);
}


std::string Scanner::ScanIdentifier() {
  std::string val;
  while (!Empty()) {
      auto c = Next();
      if (IsUCN(c)) {
        c = ScanEscaped(); // Call ScanUCN()
        AppendUCN(val, c);
      } else {
        val.push_back(c);
      }
  }
  return val;
}


Token* Scanner::SkipIdentifier() {
  PutBack();
  auto c = Next();
  while (isalnum(c)
       || (0x80 <= c && c <= 0xfd)
       || c == '_'
       || c == '$'
       || IsUCN(c)) {
    if (IsUCN(c))
      c = ScanEscaped(); // Just read it
    c = Next();
  }
  PutBack();
  return MakeToken(Token::IDENTIFIER);
}


// Scan PP-Number
Token* Scanner::SkipNumber() {
  PutBack();
  bool sawHexPrefix = false;
  int tag = Token::I_CONSTANT;
  auto c = Next();
  while (c == '.' || isdigit(c) || isalpha(c) || c == '_' || IsUCN(c)) {
    if (c == 'e' || c =='E' || c == 'p' || c == 'P') {
      if (!Try('-')) Try('+');
      if (!((c == 'e' || c == 'E') && sawHexPrefix))
        tag = Token::F_CONSTANT;
    }  else if (IsUCN(c)) {
      ScanEscaped();
    } else if (c == '.') {
      tag = Token::F_CONSTANT;
    } else if (c == 'x' || c == 'X') {
      sawHexPrefix = true;
    }
    c = Next();
  }
  PutBack();
  return MakeToken(tag);
}


Encoding Scanner::ScanLiteral(std::string& val) {
  auto enc = Test('\"') ? Encoding::NONE: ScanEncoding(Next());
  Next();
  val.resize(0);
  while (!Test('\"')) {
    auto c = Next();
    bool isucn = IsUCN(c);
    if (c == '\\')
      c = ScanEscaped();
    if (isucn)
      AppendUCN(val, c);
    else
      val.push_back(c);
  }
  return enc;
}


Token* Scanner::SkipLiteral() {
  auto c = Next();
  while (c != '\"' && c != '\n' && c != '\0') {
    if (c == '\\') Next();
    c = Next();
  }
  if (c != '\"')
    Error(loc_, "unterminated string literal");
  return MakeToken(Token::LITERAL);
}


Encoding Scanner::ScanCharacter(int& val) {
  auto enc = Test('\'') ? Encoding::NONE: ScanEncoding(Next());
  Next();
  val = 0;
  while (!Test('\'')) {
    auto c = Next();
    if (c == '\\')
      c = ScanEscaped();
    if (enc == Encoding::NONE)
      val = (val << 8) + c;
    else
      val = c;
  }
  return enc;
}


Token* Scanner::SkipCharacter() {
  auto c = Next();
  while (c != '\'' && c != '\n' && c != '\0') {
    if (c == '\\') Next();
    c = Next();
  }
  if (c != '\'')
    Error(loc_, "unterminated character constant");
  return MakeToken(Token::C_CONSTANT);
}


int Scanner::ScanEscaped() {
  auto c = Next();
  switch (c) {
  case '\\': case '\'': case '\"': case '\?':
    return c;
  case 'a': return '\a';
  case 'b': return '\b';
  case 'f': return '\f';
  case 'n': return '\n';
  case 'r': return '\r';
  case 't': return '\t';
  case 'v': return '\v';
  // Non-standard GCC extention
  case 'e': return '\033';
  case 'x': return ScanHexEscaped();
  case '0' ... '7': return ScanOctEscaped(c);
  case 'u': return ScanUCN(4);
  case 'U': return ScanUCN(8);
  default: Error(loc_, "unrecognized escape character '%c'", c);
  }
  return c; // Make compiler happy
}


int Scanner::ScanHexEscaped() {
  int val = 0, c = Peek();
  if (!isxdigit(c))
    Error(loc_, "expect xdigit, but got '%c'", c);
  while (isxdigit(c)) {
    val = (val << 4) + XDigit(c);
    Next();
    c = Peek();
  }
  return val;
}


int Scanner::ScanOctEscaped(int c) {
  int val = XDigit(c);
  c = Peek();
  if (!IsOctal(c))
    return val;
  val = (val << 3) + XDigit(c);
  Next();

  c = Peek();
  if (!IsOctal(c))
    return val;
  val = (val << 3) + XDigit(c);
  Next();
  return val;
}


int Scanner::ScanUCN(int len) {
  assert(len == 4 || len == 8);
  int val = 0;
  for (auto i = 0; i < len; ++i) {
    auto c = Next();
    if (!isxdigit(c))
      Error(loc_, "expect xdigit, but got '%c'", c);
    val = (val << 4) + XDigit(c);
  }
  return val;
}


int Scanner::XDigit(int c) {
  switch (c) {
  case '0' ... '9': return c - '0';
  case 'a' ... 'z': return c - 'a' + 10;
  case 'A' ... 'Z': return c - 'A' + 10;
  default: assert(false); return c;
  }
}


Encoding Scanner::ScanEncoding(int c) {
  switch (c) {
  case 'u': return Try('8') ? Encoding::UTF8: Encoding::CHAR16;
  case 'U': return Encoding::CHAR32;
  case 'L': return Encoding::WCHAR;
  default: assert(false); return Encoding::NONE;
  }
}


std::string* ReadFile(const std::string& filename) {
  FILE* f = fopen(filename.c_str(), "r");
  if (!f) Error("%s: No such file or directory", filename.c_str());
  auto text = new std::string;
  int c;
  while (EOF != (c = fgetc(f)))
      text->push_back(c);
  fclose(f);
  return text;
}


int Scanner::Next() {
  int c = Peek();
  ++p_;
  if (c == '\n') {
    ++loc_.line_;
    loc_.column_ = 1;
    loc_.lineBegin_ = p_;
  } else {
    ++loc_.column_;
  }
  return c;
}


int Scanner::Peek() {
  int c = (uint8_t)(*p_);
  if (c == '\\' && p_[1] == '\n') {
    p_ += 2;
    ++loc_.line_;
    loc_.column_ = 1;
    loc_.lineBegin_ = p_;
    return Peek();
  }
  return c;
}


// There couldn't be more than one PutBack() that
// cross two line, so just leave lineBegin, because
// we never care about the pos of newline token
void Scanner::PutBack() {
  int c = *--p_;
  if (c == '\n' && p_[-1] == '\\') {
    --loc_.line_;
    --p_;
    return PutBack();
  } else if (c == '\n') {
    --loc_.line_;
  } else {
    --loc_.column_;
  }
}


Token* Scanner::MakeToken(int tag) {
  tok_.tag_ = tag;
  auto& str = tok_.str_;
  str.resize(0);
  const char* p = tok_.loc_.lineBegin_ + tok_.loc_.column_ - 1;
  for (; p < p_; ++p) {
    if (p[0] == '\n' && p[-1] == '\\')
      str.pop_back();
    else
      str.push_back(p[0]);
  }
  return Token::New(tok_);
}


/*
 * New line is special, it is generated before reading the character '\n'
 */
Token* Scanner::MakeNewLine() {
  tok_.tag_ = '\n';
  tok_.str_ = std::string(p_, p_ + 1);
  return Token::New(tok_);
}
