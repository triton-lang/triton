#ifndef _WGTCC_SCANNER_H_
#define _WGTCC_SCANNER_H_

#include "error.h"
#include "encoding.h"
#include "token.h"

#include <string>
#include <cassert>


class Scanner {
public:
  explicit Scanner(const Token* tok)
      : Scanner(&tok->str_, tok->loc_) {}
  Scanner(const std::string* text, const SourceLocation& loc)
      : Scanner(text, loc.filename_, loc.line_, loc.column_) {}
  explicit Scanner(const std::string* text,
                   const std::string* filename=nullptr,
                   unsigned line=1, unsigned column=1)
      : text_(text), tok_(Token::END) {
    // TODO(wgtdkp): initialization
    p_ = &(*text_)[0];
    loc_ = {filename, p_, line, 1};
  }

  virtual ~Scanner() {}
  Scanner(const Scanner& other) = delete;
  Scanner& operator=(const Scanner& other) = delete;

  // Scan plain text and generate tokens in ts.
  // The param 'ts' need not be empty, if so, the tokens
  // are inserted at the *header* of 'ts'.
  // The param 'ws' tells if there is leading white space
  // before this token, it is only SkipComment() that will
  // set this param.
  Token* Scan(bool ws=false);
  void Tokenize(TokenSequence& ts);
  static std::string ScanHeadName(const Token* lhs, const Token* rhs);
  Encoding ScanCharacter(int& val);
  Encoding ScanLiteral(std::string& val);
  std::string ScanIdentifier();

private:
  Token* SkipIdentifier();
  Token* SkipNumber();
  Token* SkipLiteral();
  Token* SkipCharacter();
  Token* MakeToken(int tag);
  Token* MakeNewLine();
  Encoding ScanEncoding(int c);
  int ScanEscaped();
  int ScanHexEscaped();
  int ScanOctEscaped(int c);
  int ScanUCN(int len);
  void SkipWhiteSpace();
  void SkipComment();
  bool IsUCN(int c) { return c == '\\' && (Test('u') || Test('U')); }
  bool IsOctal(int c) { return '0' <= c && c <= '7'; }
  int XDigit(int c);
  bool Empty() const { return *p_ == 0; }
  int Peek();
  bool Test(int c) { return Peek() == c; };
  int Next();
  void PutBack();
  bool Try(int c) {
    if (Peek() == c) {
      Next();
      return true;
    }
    return false;
  };
  void Mark() { tok_.loc_ = loc_; };

  const std::string* text_;
  SourceLocation loc_;
  Token tok_;
  const char* p_;
};


std::string* ReadFile(const std::string& filename);

#endif
