#include "triton/lang/token.h"

#include "triton/lang/mem_pool.h"
#include "triton/lang/parser.h"


static MemPoolImp<Token> tokenPool;

const std::unordered_map<std::string, int> Token::kwTypeMap_ {
  { "auto", Token::AUTO },
  { "break", Token::BREAK },
  { "case", Token::CASE },
  { "char", Token::CHAR },
  { "const", Token::CONST },
  { "continue", Token::CONTINUE },
  { "default", Token::DEFAULT },
  { "do", Token::DO },
  { "double", Token::DOUBLE },
  { "else", Token::ELSE },
  { "enum", Token::ENUM },
  { "extern", Token::EXTERN },
  { "float", Token::FLOAT },
  { "for", Token::FOR },
  { "goto", Token::GOTO },
  { "half", Token::HALF },
  { "if", Token::IF },
  { "inline", Token::INLINE },
  { "int", Token::INT },
  { "long", Token::LONG },
  { "newaxis", Token::NEWAXIS },
  { "signed", Token::SIGNED },
  { "unsigned", Token::UNSIGNED },
  { "register", Token::REGISTER },
  { "restrict", Token::RESTRICT },
  { "return", Token::RETURN },
  { "short", Token::SHORT },
  { "sizeof", Token::SIZEOF },
  { "static", Token::STATIC },
  { "struct", Token::STRUCT },
  { "switch", Token::SWITCH },
  { "typedef", Token::TYPEDEF },
  { "union", Token::UNION },
  { "void", Token::VOID },
  { "volatile", Token::VOLATILE },
  { "while", Token::WHILE },
  { "_Alignas", Token::ALIGNAS },
  { "_Alignof", Token::ALIGNOF },
  { "_Atomic", Token::ATOMIC },
  { "__attribute__", Token::ATTRIBUTE },
  { "_Bool", Token::BOOL },
  { "_Complex", Token::COMPLEX },
  { "_Generic", Token::GENERIC },
  { "_Imaginary", Token::IMAGINARY },
  { "_Noreturn", Token::NORETURN },
  { "_Static_assert", Token::STATIC_ASSERT },
  { "_Thread_local", Token::THREAD },
};

const std::unordered_map<int, const char*> Token::tagLexemeMap_ {
  { '(', "(" },
  { ')', ")" },
  { '[', "[" },
  { ']', "]" },
  { ':', ":" },
  { ',', "," },
  { ';', ";" },
  { '+', "+" },
  { '-', "-" },
  { '*', "*" },
  { '/', "/" },
  { '|', "|" },
  { '&', "&" },
  { '<', "<" },
  { '>', ">" },
  { '=', "=" },
  { '.', "." },
  { '%', "%" },
  { '{', "{" },
  { '}', "}" },
  { '^', "^" },
  { '~', "~" },
  { '!', "!" },
  { '?', "?" },
  { '#', "#" },
  { '@', "@" },

  { Token::DSHARP, "##" },
  { Token::PTR, "->" },
  { Token::INC, "++" },
  { Token::DEC, "--" },
  { Token::LEFT, "<<" },
  { Token::RIGHT, ">>" },
  { Token::LE, "<=" },
  { Token::GE, ">=" },
  { Token::EQ, "==" },
  { Token::NE, "!=" },
  { Token::LOGICAL_AND, "&&" },
  { Token::LOGICAL_OR, "||" },
  { Token::MUL_ASSIGN, "*=" },
  { Token::DIV_ASSIGN, "/=" },
  { Token::MOD_ASSIGN, "%=" },
  { Token::ADD_ASSIGN, "+=" },
  { Token::SUB_ASSIGN, "-=" },
  { Token::LEFT_ASSIGN, "<<=" },
  { Token::RIGHT_ASSIGN, ">>=" },
  { Token::AND_ASSIGN, "&=" },
  { Token::XOR_ASSIGN, "^=" },
  { Token::OR_ASSIGN, "|=" },
  { Token::ELLIPSIS, "..." },
  { Token::AUTO, "auto" },
  { Token::BREAK, "break" },
  { Token::CASE, "case" },
  { Token::CHAR, "char" },
  { Token::CONST, "const" },
  { Token::CONTINUE, "continue" },
  { Token::DEFAULT, "default" },
  { Token::DO, "do" },
  { Token::DOUBLE, "double" },
  { Token::ELSE, "else" },
  { Token::ENUM, "enum" },
  { Token::EXTERN, "extern" },
  { Token::FLOAT, "float" },
  { Token::FOR, "for" },
  { Token::GOTO, "goto" },
  { Token::IF, "if" },
  { Token::INLINE, "inline" },
  { Token::INT, "int" },
  { Token::LONG, "long" },
  { Token::NEWAXIS, "newaxis" },
  { Token::SIGNED, "signed" },
  { Token::UNSIGNED, "unsigned" },
  { Token::REGISTER, "register" },
  { Token::RESTRICT, "restrict" },
  { Token::RETURN, "return" },
  { Token::SHORT, "short" },
  { Token::SIZEOF, "sizeof" },
  { Token::STATIC, "static" },
  { Token::STRUCT, "struct" },
  { Token::SWITCH, "switch" },
  { Token::TYPEDEF, "typedef" },
  { Token::UNION, "union" },
  { Token::VOID, "void" },
  { Token::VOLATILE, "volatile" },
  { Token::WHILE, "while" },
  { Token::ALIGNAS, "_Alignas" },
  { Token::ALIGNOF, "_Alignof" },
  { Token::ATOMIC, "_Atomic" },
  { Token::ATTRIBUTE, "__attribute__" },
  { Token::BOOL, "_Bool" },
  { Token::COMPLEX, "_Complex" },
  { Token::GENERIC, "_Generic" },
  { Token::IMAGINARY, "_Imaginary" },
  { Token::NORETURN, "_Noreturn" },
  { Token::STATIC_ASSERT, "_Static_assert" },
  { Token::THREAD, "_Thread_local" },

  { Token::END, "(eof)" },
  { Token::IDENTIFIER, "(identifier)" },
  { Token::CONSTANT, "(constant)" },
  { Token::LITERAL, "(string literal)" },
};


Token* Token::New(int tag) {
  return new (tokenPool.Alloc()) Token(tag);
}


Token* Token::New(const Token& other) {
  return new (tokenPool.Alloc()) Token(other);
}


Token* Token::New(int tag,
                  const SourceLocation& loc,
                  const std::string& str,
                  bool ws) {
  return new (tokenPool.Alloc()) Token(tag, loc, str, ws);
}


TokenSequence TokenSequence::GetLine() {
  auto begin = begin_;
  while (begin_ != end_ && (*begin_)->tag_ != Token::NEW_LINE)
    ++begin_;
  auto end = begin_;
  return {tokList_, begin, end};
}


/*
 * If this seq starts from the begin of a line.
 * Called only after we have saw '#' in the token sequence.
 */
bool TokenSequence::IsBeginOfLine() const {
  if (begin_ == tokList_->begin())
    return true;

  auto pre = begin_;
  --pre;

  // We do not insert a newline at the end of a source file.
  // Thus if two token have different filename, the second is
  // the begin of a line.
  return ((*pre)->tag_ == Token::NEW_LINE ||
          (*pre)->loc_.filename_ != (*begin_)->loc_.filename_);
}

const Token* TokenSequence::Peek() const {
  static auto eof = Token::New(Token::END);
  if (begin_ != end_ && (*begin_)->tag_ == Token::NEW_LINE) {
    ++begin_;
    return Peek();
  } else if (begin_ == end_) {
    if (end_ != tokList_->begin())
      *eof = *Back();
    eof->tag_ = Token::END;
    return eof;
  } else if (parser_ && (*begin_)->tag_ == Token::IDENTIFIER &&
             (*begin_)->str_ == "__func__") {
    auto filename = Token::New(*(*begin_));
    filename->tag_ = Token::LITERAL;
    filename->str_ = "\"" + parser_->CurFunc()->Name() + "\"";
    *begin_ = filename;
  }
  return *begin_;
}


const Token* TokenSequence::Expect(int expect) {
  auto tok = Peek();
  if (!Try(expect)) {
    Error(tok, "'%s' expected, but got '%s'",
        Token::Lexeme(expect), tok->str_.c_str());
  }
  return tok;
}

void TokenSequence::Print(FILE* fp) const {
  unsigned lastLine = 0;
  auto ts = *this;
  while (!ts.Empty()) {
    auto tok = ts.Next();
    if (lastLine != tok->loc_.line_) {
      fputs("\n", fp);
      for (unsigned i = 0; i < tok->loc_.column_; ++i)
        fputc(' ', fp);
    } else if (tok->ws_) {
      fputc(' ', fp);
    }
    fputs(tok->str_.c_str(), fp);
    fflush(fp);
    lastLine = tok->loc_.line_;
  }
  fputs("\n", fp);
}

//void TokenSequence::Print(std::string *str) const {

//}
