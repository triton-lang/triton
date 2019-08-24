#pragma once

#ifndef _WGTCC_TOKEN_H_
#define _WGTCC_TOKEN_H_

#include "error.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <list>
#include <set>
#include <string>
#include <unordered_map>


class Generator;
class Parser;
class Scanner;
class Token;
class TokenSequence;

using HideSet = std::set<std::string>;
using TokenList = std::list<const Token*>;


struct SourceLocation {
  const std::string* filename_;
  const char* lineBegin_;
  unsigned line_;
  unsigned column_;

  const char* Begin() const {
    return lineBegin_ + column_ - 1;
  }
};


class Token {
  friend class Scanner;
public:
  enum {
    // Punctuators
    LPAR = '(',
    RPAR = ')',
    LSQB = '[',
    RSQB = ']',
    COLON = ':',
    COMMA = ',',
    SEMI = ';',
    ADD = '+',
    SUB = '-',
    MUL = '*',
    DIV = '/',
    OR = '|',
    AND = '&',
    XOR = '^',
    LESS = '<',
    GREATER = '>',
    EQUAL = '=',
    DOT = '.',
    MOD = '%',
    LBRACE = '{',
    RBRACE = '}',
    TILDE = '~',
    NOT = '!',
    COND = '?',
    SHARP = '#',
    MATMUL = '@',
    NEW_LINE = '\n',

    DSHARP = 128, // '##'
    PTR,
    INC,
    DEC,
    LEFT,
    RIGHT,
    LE,
    GE,
    EQ,
    NE,
    LOGICAL_AND,
    LOGICAL_OR,

    MUL_ASSIGN,
    DIV_ASSIGN,
    MOD_ASSIGN,
    ADD_ASSIGN,
    SUB_ASSIGN,
    LEFT_ASSIGN,
    RIGHT_ASSIGN,
    AND_ASSIGN,
    XOR_ASSIGN,
    OR_ASSIGN,

    ELLIPSIS,
    // Punctuators end

    // KEYWORD BEGIN
    // TYPE QUALIFIER BEGIN
    CONST,
    RESTRICT,
    VOLATILE,
    ATOMIC,
    // TYPE QUALIFIER END

    // TYPE SPECIFIER BEGIN
    VOID,
    CHAR,
    SHORT,
    INT,
    LONG,
    HALF,
    FLOAT,
    DOUBLE,
    SIGNED,
    UNSIGNED,
    BOOL,		// _Bool
    COMPLEX,	// _Complex
    STRUCT,
    UNION,
    ENUM,
    // TYPE SPECIFIER END

    ATTRIBUTE, // GNU extension __attribute__
    // FUNCTION SPECIFIER BEGIN
    INLINE,
    NORETURN,	// _Noreturn
    // FUNCTION SPECIFIER END

    // TILE ARITHMETICS BEGIN
    NEWAXIS,
    // TILE ARITHMETICS END

    ALIGNAS, // _Alignas
    // For syntactic convenience
    STATIC_ASSERT, // _Static_assert
    // STORAGE CLASS SPECIFIER BEGIN
    TYPEDEF,
    EXTERN,
    STATIC,
    THREAD,	// _Thread_local
    AUTO,
    REGISTER,

    // STORAGE CLASS SPECIFIER END
    BREAK,
    CASE,
    CONTINUE,
    DEFAULT,
    DO,
    ELSE,
    FOR,
    GOTO,
    IF,
    RETURN,
    SIZEOF,
    SWITCH,
    WHILE,
    ALIGNOF, // _Alignof
    GENERIC, // _Generic
    IMAGINARY, // _Imaginary
    // KEYWORD END

    IDENTIFIER,
    CONSTANT,
    I_CONSTANT,
    C_CONSTANT,
    F_CONSTANT,
    LITERAL,

    // For the parser, a identifier is a typedef name or user defined type
    POSTFIX_INC,
    POSTFIX_DEC,
    PREFIX_INC,
    PREFIX_DEC,
    ADDR,  // '&'
    DEREF, // '*'
    PLUS,
    MINUS,
    CAST,

    // For preprocessor
    PP_IF,
    PP_IFDEF,
    PP_IFNDEF,
    PP_ELIF,
    PP_ELSE,
    PP_ENDIF,
    PP_INCLUDE,
    PP_DEFINE,
    PP_UNDEF,
    PP_LINE,
    PP_ERROR,
    PP_PRAGMA,
    PP_NONE,
    PP_EMPTY,


    IGNORE,
    INVALID,
    END,
    NOTOK = -1,
  };

  static Token* New(int tag);
  static Token* New(const Token& other);
  static Token* New(int tag,
                    const SourceLocation& loc,
                    const std::string& str,
                    bool ws=false);
  Token& operator=(const Token& other) {
    tag_ = other.tag_;
    ws_ = other.ws_;
    loc_ = other.loc_;
    str_ = other.str_;
    hs_ = other.hs_ ? new HideSet(*other.hs_): nullptr;
    return *this;
  }
  virtual ~Token() {}

  // Token::NOTOK represents not a kw.
  static int KeyWordTag(const std::string& key) {
    auto kwIter = kwTypeMap_.find(key);
    if (kwTypeMap_.end() == kwIter)
      return Token::NOTOK;	// Not a key word type
    return kwIter->second;
  }
  static bool IsKeyWord(const std::string& name);
  static bool IsKeyWord(int tag) { return CONST <= tag && tag < IDENTIFIER; }
  bool IsKeyWord() const { return IsKeyWord(tag_); }
  bool IsPunctuator() const { return 0 <= tag_ && tag_ <= ELLIPSIS; }
  bool IsLiteral() const { return tag_ == LITERAL; }
  bool IsConstant() const { return CONSTANT <= tag_ && tag_ <= F_CONSTANT; }
  bool IsIdentifier() const { return IDENTIFIER == tag_; }
  bool IsEOF() const { return tag_ == Token::END; }
  bool IsTypeSpecQual() const { return CONST <= tag_ && tag_ <= ENUM; }
  bool IsDecl() const { return CONST <= tag_ && tag_ <= REGISTER; }
  static const char* Lexeme(int tag) {
    auto iter = tagLexemeMap_.find(tag);
    if (iter == tagLexemeMap_.end())
      return nullptr;

    return iter->second;
  }

  int tag_;

  // 'ws_' standards for weither there is preceding white space
  // This is to simplify the '#' operator(stringize) in macro expansion
  bool ws_ { false };
  SourceLocation loc_;

  std::string str_;
  HideSet* hs_ { nullptr };

private:
  explicit Token(int tag): tag_(tag) {}
  Token(int tag, const SourceLocation& loc,
        const std::string& str, bool ws=false)
      : tag_(tag), ws_(ws), loc_(loc), str_(str) {}

  Token(const Token& other) {
    *this = other;
  }

  static const std::unordered_map<std::string, int> kwTypeMap_;
  static const std::unordered_map<int, const char*> tagLexemeMap_;
};


class TokenSequence {
  friend class Preprocessor;

public:
  TokenSequence(): tokList_(new TokenList()),
                   begin_(tokList_->begin()), end_(tokList_->end()) {}
  explicit TokenSequence(Token* tok) {
    TokenSequence();
    InsertBack(tok);
  }
  explicit TokenSequence(TokenList* tokList)
      : tokList_(tokList),
        begin_(tokList->begin()),
        end_(tokList->end()) {}
  TokenSequence(TokenList* tokList,
                TokenList::iterator begin,
                TokenList::iterator end)
      : tokList_(tokList), begin_(begin), end_(end) {}
  ~TokenSequence() {}
  TokenSequence(const TokenSequence& other) { *this = other; }
  const TokenSequence& operator=(const TokenSequence& other) {
    tokList_ = other.tokList_;
    begin_ = other.begin_;
    end_ = other.end_;
    return *this;
  }
  void Copy(const TokenSequence& other) {
    tokList_ = new TokenList(other.begin_, other.end_);
    begin_ = tokList_->begin();
    end_ = tokList_->end();
    for (auto iter = begin_; iter != end_; ++iter)
      *iter = Token::New(**iter);
  }
  void UpdateHeadLocation(const SourceLocation& loc) {
    assert(!Empty());
    auto tok = const_cast<Token*>(Peek());
    tok->loc_ = loc;
  }
  void FinalizeSubst(bool leadingWS, const HideSet& hs) {
    auto ts = *this;
    while (!ts.Empty()) {
      auto tok = const_cast<Token*>(ts.Next());
      if (!tok->hs_)
        tok->hs_ = new HideSet(hs);
      else
        tok->hs_->insert(hs.begin(), hs.end());
    }
    // Even if the token sequence is empty
    const_cast<Token*>(Peek())->ws_ = leadingWS;
  }

  const Token* Expect(int expect);
  bool Try(int tag) {
    if (Peek()->tag_ == tag) {
      Next();
      return true;
    }
    return false;
  }
  bool Test(int tag) { return Peek()->tag_ == tag; }
  const Token* Next() {
    auto ret = Peek();
    if (!ret->IsEOF()) {
      ++begin_;
      Peek(); // May skip newline token, but why ?
    } else {
      ++exceed_end;
    }
    return ret;
  }
  void PutBack() {
    assert(begin_ != tokList_->begin());
    if (exceed_end > 0) {
      --exceed_end;
    } else {
      --begin_;
      if ((*begin_)->tag_ == Token::NEW_LINE)
        PutBack();
    }
  }
  const Token* Peek() const;
  const Token* Peek2() {
    if (Empty())
      return Peek(); // Return the Token::END
    Next();
    auto ret = Peek();
    PutBack();
    return ret;
  }
  const Token* Back() const {
    auto back = end_;
    return *--back;
  }
  void PopBack() {
    assert(!Empty());
    assert(end_ == tokList_->end());
    auto size_eq1 = tokList_->back() == *begin_;
    tokList_->pop_back();
    end_ = tokList_->end();
    if (size_eq1)
      begin_ = end_;
  }
  TokenList::iterator Mark() { return begin_; }
  void ResetTo(TokenList::iterator mark) { begin_ = mark; }
  bool Empty() const { return Peek()->tag_ == Token::END; }
  void InsertBack(TokenSequence& ts) {
    auto pos = tokList_->insert(end_, ts.begin_, ts.end_);
    if (begin_ == end_) {
      begin_ = pos;
    }
  }
  void InsertBack(const Token* tok) {
    auto pos = tokList_->insert(end_, tok);
    if (begin_ == end_) {
      begin_ = pos;
    }
  }

  // If there is preceding newline
  void InsertFront(TokenSequence& ts) {
    auto pos = GetInsertFrontPos();
    begin_ = tokList_->insert(pos, ts.begin_, ts.end_);
  }
  void InsertFront(const Token* tok) {
    auto pos = GetInsertFrontPos();
    begin_ = tokList_->insert(pos, tok);
  }
  bool IsBeginOfLine() const;
  TokenSequence GetLine();
  void SetParser(Parser* parser) { parser_ = parser; }
  void Print(FILE* fp=stdout) const;
  void Print(std::string *str) const;

private:
  // Find a insert position with no preceding newline
  TokenList::iterator GetInsertFrontPos() {
    auto pos = begin_;
    if (pos == tokList_->begin())
      return pos;
    --pos;
    while (pos != tokList_->begin() && (*pos)->tag_ == Token::NEW_LINE)
      --pos;
    return ++pos;
  }

  TokenList* tokList_;
  mutable TokenList::iterator begin_;
  TokenList::iterator end_;
  Parser* parser_ {nullptr};
  int exceed_end {0};
};

#endif
