#include "triton/lang/wgtcc/cpp.h"

#include "triton/lang/wgtcc/evaluator.h"
#include "triton/lang/wgtcc/parser.h"

#include <ctime>
#include <fcntl.h>
#include <unistd.h>
#include <unordered_map>


extern std::string filename_in;
extern std::string filename_out;

using DirectiveMap = std::unordered_map<std::string, int>;

static const DirectiveMap directiveMap {
  {"if", Token::PP_IF},
  {"ifdef", Token::PP_IFDEF},
  {"ifndef", Token::PP_IFNDEF},
  {"elif", Token::PP_ELIF},
  {"else", Token::PP_ELSE},
  {"endif", Token::PP_ENDIF},
  {"include", Token::PP_INCLUDE},
  // Non-standard GNU extension
  {"include_next", Token::PP_INCLUDE},
  {"define", Token::PP_DEFINE},
  {"undef", Token::PP_UNDEF},
  {"line", Token::PP_LINE},
  {"error", Token::PP_ERROR},
  {"pragma", Token::PP_PRAGMA}
};


/*
 * params:
 *  is: input token sequence
 *  os: output token sequence
 */
void Preprocessor::Expand(TokenSequence& os, TokenSequence is, bool inCond) {
  Macro* macro = nullptr;
  int direcitve;
  while (!is.Empty()) {
    UpdateFirstTokenLine(is);
    auto tok = is.Peek();
    const auto& name = tok->str_;

    if ((direcitve = GetDirective(is)) != Token::INVALID) {
      ParseDirective(os, is, direcitve);
    } else if (!inCond && !NeedExpand()) {
      // Discards the token
      is.Next();
    } else if (inCond && name == "defined") {
      is.Next();
      os.InsertBack(EvalDefOp(is));
    } else if (tok->hs_ && tok->hs_->find(name) != tok->hs_->end()) {
      os.InsertBack(is.Next());
    } else if ((macro = FindMacro(name))) {
      is.Next();

      if (name == "__FILE__") {
        HandleTheFileMacro(os, tok);
      } else if (name == "__LINE__") {
        HandleTheLineMacro(os, tok);
      } else if (macro->ObjLike()) {
        // Make a copy, as subst will change repSeq
        auto repSeq = macro->RepSeq(tok->loc_.filename_, tok->loc_.line_);

        TokenList tokList;
        TokenSequence repSeqSubsted(&tokList);
        ParamMap paramMap;
        // TODO(wgtdkp): hideset is not right
        // Make a copy of hideset
        // HS U {name}
        auto hs = tok->hs_ ? *tok->hs_: HideSet();
        hs.insert(name);
        Subst(repSeqSubsted, repSeq, tok->ws_, hs, paramMap);
        is.InsertFront(repSeqSubsted);
      } else if (is.Try('(')) {
        ParamMap paramMap;
        auto rpar = ParseActualParam(is, macro, paramMap);
        auto repSeq = macro->RepSeq(tok->loc_.filename_, tok->loc_.line_);
        TokenList tokList;
        TokenSequence repSeqSubsted(&tokList);

        // (HS ^ HS') U {name}
        // Use HS' U {name} directly
        auto hs = rpar->hs_ ? *rpar->hs_: HideSet();
        hs.insert(name);
        Subst(repSeqSubsted, repSeq, tok->ws_, hs, paramMap);
        is.InsertFront(repSeqSubsted);
      } else {
        os.InsertBack(tok);
      }
    } else {
      os.InsertBack(is.Next());
    }
  }
}


static bool FindActualParam(TokenSequence& ap,
                            ParamMap& params,
                            const std::string& fp) {
  auto res = params.find(fp);
  if (res == params.end()) {
    return false;
  }
  ap.Copy(res->second);
  return true;
}


void Preprocessor::Subst(TokenSequence& os,
                         TokenSequence is,
                         bool leadingWS,
                         const HideSet& hs,
                         ParamMap& params) {
  TokenSequence ap;

  while (!is.Empty()) {
    if (is.Test('#') && FindActualParam(ap, params, is.Peek2()->str_)) {
      is.Next(); is.Next();
      auto tok = Stringize(ap);
      os.InsertBack(tok);
    } else if (is.Test(Token::DSHARP) &&
               FindActualParam(ap, params, is.Peek2()->str_)) {
      is.Next(); is.Next();
      if (!ap.Empty())
        Glue(os, ap);
    } else if (is.Test(Token::DSHARP)) {
      is.Next();
      auto tok = is.Next();
      Glue(os, tok);
    } else if (is.Peek2()->tag_ == Token::DSHARP &&
               FindActualParam(ap, params, is.Peek()->str_)) {
      is.Next();

      if (ap.Empty()) {
        is.Next();
        if (FindActualParam(ap, params, is.Peek()->str_)) {
          is.Next();
          os.InsertBack(ap);
        }
      } else {
        os.InsertBack(ap);
      }
    } else if (FindActualParam(ap, params, is.Peek()->str_)) {
      auto tok = is.Next();
      const_cast<Token*>(ap.Peek())->ws_ = tok->ws_;
      Expand(os, ap);
    } else {
      os.InsertBack(is.Peek());
      is.Next();
    }
  }

  os.FinalizeSubst(leadingWS, hs);
}


void Preprocessor::Glue(TokenSequence& os, const Token* tok) {
  TokenList tokList {tok};
  TokenSequence is(&tokList);
  Glue(os, is);
}


void Preprocessor::Glue(TokenSequence& os, TokenSequence is) {
  auto lhs = os.Back();
  auto rhs = is.Peek();

  auto str = new std::string(lhs->str_ + rhs->str_);
  TokenSequence ts;
  Scanner scanner(str, lhs->loc_);
  scanner.Tokenize(ts);

  is.Next();

  if (ts.Empty()) {
    // TODO(wgtdkp):
    // No new Token generated
    // How to handle it???
  } else {
    os.PopBack();
    auto newTok = const_cast<Token*>(ts.Next());
    newTok->ws_ = lhs->ws_;
    newTok->hs_ = lhs->hs_;
    os.InsertBack(newTok);
  }

  if (!ts.Empty()) {
    Error(lhs, "macro expansion failed: cannot concatenate");
  }

  os.InsertBack(is);
}


/*
 * This is For the '#' operator in func-like macro
 */
const Token* Preprocessor::Stringize(TokenSequence is) {
  std::string str = "\"";
  while (!is.Empty()) {
    auto tok = is.Next();
    // Have preceding white space
    // and is not the first token of the sequence
    str.append(tok->ws_ && str.size() > 1, ' ');
    if (tok->tag_ == Token::LITERAL || tok->tag_ == Token::C_CONSTANT) {
      for (auto c: tok->str_) {
        if (c == '"' || c == '\\')
          str.push_back('\\');
        str.push_back(c);
      }
    } else {
      str += tok->str_;
    }
  }
  str.push_back('\"');

  auto ret = Token::New(*is.Peek());
  ret->tag_ = Token::LITERAL;
  ret->str_ = str;
  return ret;
}


void Preprocessor::Finalize(TokenSequence os) {
  while (!os.Empty()) {
    auto tok = os.Next();
    if (tok->tag_ == Token::INVALID) {
      Error(tok, "stray token in program");
    } else if (tok->tag_ == Token::IDENTIFIER) {
      auto tag = Token::KeyWordTag(tok->str_);
      if (Token::IsKeyWord(tag)) {
        const_cast<Token*>(tok)->tag_ = tag;
      } else {
        const_cast<Token*>(tok)->str_ = Scanner(tok).ScanIdentifier();
      }
    }
    if (fName_ && !tok->loc_.filename_) {
      assert(false);
    }
  }
}


// TODO(wgtdkp): add predefined macros
void Preprocessor::Process(TokenSequence& os) {
  TokenSequence is;
  // Add source file
  if(fName_)
    IncludeFile(is, fName_);
  else
    IncludeSrc(is, fSrc_, nullptr);
  // Expand
  Expand(os, is);
  Finalize(os);
}


const Token* Preprocessor::ParseActualParam(TokenSequence& is,
                                            Macro* macro,
                                            ParamMap& paramMap) {
  const Token* ret;
  if (macro->Params().size() == 0 && !macro->Variadic()) {
    ret = is.Next();
    if (ret->tag_ != ')')
      Error(ret, "too many arguments");
    return ret;
  }

  auto fp = macro->Params().begin();
  TokenSequence ap;

  int cnt = 1;
  while (cnt > 0) {
    if (is.Empty())
      Error(is.Peek(), "premature end of input");
    else if (is.Test('('))
      ++cnt;
    else if (is.Test(')'))
      --cnt;

    if ((is.Test(',') && cnt == 1) || cnt == 0) {

      if (fp == macro->Params().end()) {
        if (!macro->Variadic())
          Error(is.Peek(), "too many arguments");
        if (cnt == 0)
          paramMap.insert(std::make_pair("__VA_ARGS__", ap));
        else
          ap.InsertBack(is.Peek());
      } else {
        paramMap.insert(std::make_pair(*fp, ap));
        ap = TokenSequence();
        ++fp;
      }
    } else {
      ap.InsertBack(is.Peek());
    }
    ret = is.Next();
  }

  if (fp != macro->Params().end())
    Error(is.Peek(), "too few params");
  return ret;
}


const Token* Preprocessor::EvalDefOp(TokenSequence& is) {
  auto hasPar = is.Try('(');
  auto macro = is.Expect(Token::IDENTIFIER);
  auto cons = Token::New(*macro);
  if (hasPar) is.Expect(')');
  cons->tag_ = Token::I_CONSTANT;
  cons->str_ = FindMacro(macro->str_) ? "1": "0";
  return cons;
}


void Preprocessor::ReplaceIdent(TokenSequence& is) {
  TokenSequence os;
  while (!is.Empty()) {
    auto tok = is.Next();
    if (tok->tag_ == Token::IDENTIFIER) {
      auto cons = Token::New(*tok);
      cons->tag_ = Token::I_CONSTANT;
      cons->str_ = "0";
      os.InsertBack(cons);
    } else {
      os.InsertBack(tok);
    }
  }
  is = os;
}


int Preprocessor::GetDirective(TokenSequence& is) {
  if (!is.Test('#') || !is.IsBeginOfLine())
    return Token::INVALID;

  is.Next();
  if (is.IsBeginOfLine())
    return Token::PP_EMPTY;

  auto tag = is.Peek()->tag_;
  if (tag == Token::IDENTIFIER || Token::IsKeyWord(tag)) {
    auto str = is.Peek()->str_;
    auto res = directiveMap.find(str);
    if (res == directiveMap.end())
      return Token::PP_NONE;
    return res->second;
  }
  return Token::PP_NONE;
}


void Preprocessor::ParseDirective(TokenSequence& os,
                                  TokenSequence& is,
                                  int directive) {
  if (directive == Token::PP_EMPTY)
    return;
  auto ls = is.GetLine();
  switch(directive) {
  case Token::PP_IF:
    ParseIf(ls); break;
  case Token::PP_IFDEF:
    ParseIfdef(ls); break;
  case Token::PP_IFNDEF:
    ParseIfndef(ls); break;
  case Token::PP_ELIF:
    ParseElif(ls); break;
  case Token::PP_ELSE:
    ParseElse(ls); break;
  case Token::PP_ENDIF:
    ParseEndif(ls); break;
  case Token::PP_INCLUDE:
    if (NeedExpand())
      ParseInclude(is, ls);
    break;
  case Token::PP_DEFINE:
    if (NeedExpand())
      ParseDef(ls);
    break;
  case Token::PP_UNDEF:
    if (NeedExpand())
      ParseUndef(ls);
    break;
  case Token::PP_LINE:
    if (NeedExpand())
      ParseLine(ls);
    break;
  case Token::PP_ERROR:
    if (NeedExpand())
      ParseError(ls);
    break;
  case Token::PP_PRAGMA:
    if (NeedExpand())
      ParsePragma(ls);
    break;
  case Token::PP_NONE:
    break;
  default:
    assert(false);
  }
}


void Preprocessor::ParsePragma(TokenSequence ls) {
  // TODO(wgtdkp):
  ls.Next();
}


void Preprocessor::ParseError(TokenSequence ls) {
  ls.Next();
  const auto& literal = Stringize(ls);
  std::string msg;
  Scanner(literal).ScanLiteral(msg);
  Error(ls.Peek(), "%s", msg.c_str());
}


void Preprocessor::ParseLine(TokenSequence ls) {
  auto directive = ls.Next(); // Skip directive 'line'
  TokenSequence ts;
  Expand(ts, ls);
  auto tok = ts.Expect(Token::I_CONSTANT);

  int line = 0;
  size_t end = 0;
  try {
    line = stoi(tok->str_, &end, 10);
  } catch (const std::out_of_range& oor) {
    Error(tok, "line number out of range");
  }
  if (line == 0 || end != tok->str_.size()) {
    Error(tok, "illegal line number");
  }

  curLine_ = line;
  lineLine_ = directive->loc_.line_;
  if (ts.Empty())
    return;
  tok = ts.Expect(Token::LITERAL);

  // Enusure "s-char-sequence"
  if (tok->str_.front() != '"' || tok->str_.back() != '"') {
    Error(tok, "expect s-char-sequence");
  }
}


void Preprocessor::ParseIf(TokenSequence ls) {
  if (!NeedExpand()) {
    ppCondStack_.push({Token::PP_IF, false, false});
    return;
  }

  auto tok = ls.Next(); // Skip the directive

  if (ls.Empty()) {
    Error(tok, "expect expression in 'if' directive");
  }

  TokenSequence ts;
  Expand(ts, ls, true);
  ReplaceIdent(ts);

  Parser parser(ts);
  auto expr = parser.ParseExpr();
  if (!parser.ts().Empty()) {
    Error(parser.ts().Peek(), "unexpected extra expression");
  }
  bool cond;
  if (expr->Type()->IsFloat()) {
    cond = static_cast<bool>(Evaluator<double>().Eval(expr));
  } else {
    cond = static_cast<bool>(Evaluator<long>().Eval(expr));
  }
  ppCondStack_.push({Token::PP_IF, NeedExpand(), cond});
}


void Preprocessor::ParseIfdef(TokenSequence ls) {
  if (!NeedExpand()) {
    ppCondStack_.push({Token::PP_IFDEF, false, false});
    return;
  }

  ls.Next();
  auto ident = ls.Expect(Token::IDENTIFIER);
  if (!ls.Empty()) {
    Error(ls.Peek(), "expect new line");
  }

  auto cond = FindMacro(ident->str_) != nullptr;
  ppCondStack_.push({Token::PP_IFDEF, NeedExpand(), cond});
}


void Preprocessor::ParseIfndef(TokenSequence ls) {
  ParseIfdef(ls);
  auto top = ppCondStack_.top();
  ppCondStack_.pop();
  top.tag_ = Token::PP_IFNDEF;
  top.cond_ = !top.cond_;

  ppCondStack_.push(top);
}


void Preprocessor::ParseElif(TokenSequence ls) {
  auto directive = ls.Next(); // Skip the directive

  if (ppCondStack_.empty())
    Error(directive, "unexpected 'elif' directive");
  auto top = ppCondStack_.top();
  if (top.tag_ == Token::PP_ELSE)
    Error(directive, "unexpected 'elif' directive");

  while (!ppCondStack_.empty()) {
    top = ppCondStack_.top();
    if (top.tag_ == Token::PP_IF ||
        top.tag_ == Token::PP_IFDEF ||
        top.tag_ == Token::PP_IFNDEF ||
        top.cond_) {
      break;
    }
    ppCondStack_.pop();
  }
  if (ppCondStack_.empty())
    Error(directive, "unexpected 'elif' directive");
  auto enabled = top.enabled_;
  if (!enabled) {
    ppCondStack_.push({Token::PP_ELIF, false, false});
    return;
  }

  if (ls.Empty()) {
    Error(ls.Peek(), "expect expression in 'elif' directive");
  }

  TokenSequence ts;
  Expand(ts, ls, true);
  ReplaceIdent(ts);

  Parser parser(ts);
  auto expr = parser.ParseExpr();
  if (!parser.ts().Empty()) {
    Error(parser.ts().Peek(), "unexpected extra expression");
  }
  bool cond;
  if (expr->Type()->IsFloat()) {
    std::cout << Evaluator<double>().Eval(expr) << std::endl;
    cond = static_cast<bool>(Evaluator<double>().Eval(expr));
  } else {
    cond = static_cast<bool>(Evaluator<long>().Eval(expr));
  }
  cond = cond && !top.cond_;
  ppCondStack_.push({Token::PP_ELIF, true, cond});
}


void Preprocessor::ParseElse(TokenSequence ls) {
  auto directive = ls.Next();
  if (!ls.Empty())
    Error(ls.Peek(), "expect new line");

  if (ppCondStack_.empty())
    Error(directive, "unexpected 'else' directive");
  auto top = ppCondStack_.top();
  if (top.tag_ == Token::PP_ELSE)
    Error(directive, "unexpected 'else' directive");

  while (!ppCondStack_.empty()) {
    top = ppCondStack_.top();
    if (top.tag_ == Token::PP_IF ||
        top.tag_ == Token::PP_IFDEF ||
        top.tag_ == Token::PP_IFNDEF ||
        top.cond_) {
      break;
    }
    ppCondStack_.pop();
  }
  if (ppCondStack_.empty())
    Error(directive, "unexpected 'else' directive");

  auto cond = !top.cond_;
  auto enabled = top.enabled_;
  ppCondStack_.push({Token::PP_ELSE, enabled, cond});
}


void Preprocessor::ParseEndif(TokenSequence ls) {
  auto directive = ls.Next();
  if (!ls.Empty())
    Error(ls.Peek(), "expect new line");

  while ( !ppCondStack_.empty()) {
    auto top = ppCondStack_.top();
    ppCondStack_.pop();

    if (top.tag_ == Token::PP_IF
        || top.tag_ == Token::PP_IFDEF
        || top.tag_ == Token::PP_IFNDEF) {
      return;
    }
  }

  if (ppCondStack_.empty())
    Error(directive, "unexpected 'endif' directive");
}


// Have Read the '#'
void Preprocessor::ParseInclude(TokenSequence& is, TokenSequence ls) {
  bool next = ls.Next()->str_ == "include_next"; // Skip 'include'
  if (!ls.Test(Token::LITERAL) && !ls.Test('<')) {
    TokenSequence ts;
    Expand(ts, ls, true);
    ls = ts;
  }

  auto tok = ls.Next();
  if (tok->tag_ == Token::LITERAL) {
    if (!ls.Empty()) {
      Error(ls.Peek(), "expect new line");
    }
    std::string filename;
    Scanner(tok).ScanLiteral(filename);
    auto fullPath = SearchFile(filename, false, next, *tok->loc_.filename_);
    if (fullPath == nullptr)
      Error(tok, "%s: No such file or directory", filename.c_str());

    IncludeFile(is, fullPath);
  } else if (tok->tag_ == '<') {
    auto lhs = tok;
    auto rhs = tok;
    int cnt = 1;
    while (!(rhs = ls.Next())->IsEOF()) {
      if (rhs->tag_ == '<')
        ++cnt;
      else if (rhs->tag_ == '>')
        --cnt;
      if (cnt == 0)
        break;
    }
    if (cnt != 0)
      Error(rhs, "expect '>'");
    if (!ls.Empty())
      Error(ls.Peek(), "expect new line");

    const auto& filename = Scanner::ScanHeadName(lhs, rhs);
    auto fullPath = SearchFile(filename, true, next, *tok->loc_.filename_);
    if (fullPath == nullptr) {
      Error(tok, "%s: No such file or directory", filename.c_str());
    }
    IncludeFile(is, fullPath);
  } else {
    Error(tok, "expect filename(string or in '<>')");
  }
}


void Preprocessor::ParseUndef(TokenSequence ls) {
  ls.Next(); // Skip directive

  auto ident = ls.Expect(Token::IDENTIFIER);
  if (!ls.Empty())
    Error(ls.Peek(), "expect new line");

  RemoveMacro(ident->str_);
}


void Preprocessor::ParseDef(TokenSequence ls) {
  ls.Next();
  auto ident = ls.Expect(Token::IDENTIFIER);
  if (ident->str_ == "defined") {
    Error(ident, "'defined' cannot be used as a macro name");
  }
  auto tok = ls.Peek();
  if (tok->tag_ == '(' && !tok->ws_) {
    // There is no white space between ident and '('
    // Hence, we are defining function-like macro

    // Parse Identifier list
    ls.Next(); // Skip '('
    ParamList params;
    auto variadic = ParseIdentList(params, ls);
    const auto& macro = Macro(variadic, params, ls);
    AddMacro(ident->str_, macro);
  } else {
    AddMacro(ident->str_, Macro(ls));
  }
}


bool Preprocessor::ParseIdentList(ParamList& params, TokenSequence& is) {
  const Token* tok = is.Peek();
  while (!is.Empty()) {
    tok = is.Next();
    if (tok->tag_ == ')') {
      return false;
    } else if (tok->tag_ == Token::ELLIPSIS) {
      is.Expect(')');
      return true;
    } else if (tok->tag_ != Token::IDENTIFIER) {
      Error(tok, "expect identifier");
    }

    for (const auto& param: params) {
      if (param == tok->str_)
        Error(tok, "duplicated param");
    }
    params.push_back(tok->str_);

    if (!is.Try(',')) {
      is.Expect(')');
      return false;
    }
  }

  Error(tok, "unexpected end of line");
}

void Preprocessor::IncludeSrc(TokenSequence& is,
                              const std::string* text,
                              const std::string* filename) {
  TokenSequence ts {is.tokList_, is.begin_, is.begin_};
  Scanner scanner(text, filename);
  scanner.Tokenize(ts);

  // We done including header file
  is.begin_ = ts.begin_;
}

void Preprocessor::IncludeFile(TokenSequence& is,
                               const std::string* filename) {
  IncludeSrc(is, ReadFile(*filename), filename);
}


static std::string GetDir(const std::string& path) {
  auto pos = path.rfind('/');
  if (pos == std::string::npos)
    return "./";
  return path.substr(0, pos + 1);
}


std::string* Preprocessor::SearchFile(const std::string& name,
                                      const bool libHeader,
                                      bool next,
                                      const std::string& curPath) {
  if (libHeader && !next) {
    searchPaths_.push_back(GetDir(curPath));
  } else {
    searchPaths_.push_front(GetDir(curPath));
  }

  auto iter = searchPaths_.begin();
  for (; iter != searchPaths_.end(); ++iter) {
    auto dd = open(iter->c_str(), O_RDONLY);
    if (dd == -1) // TODO(wgtdkp): or ensure it before preprocessing
      continue;
    auto fd = openat(dd, name.c_str(), O_RDONLY);
    close(dd);
    if (fd != -1) {
      // Intentional, so that recursive include
      // will result in running out of file descriptor
      //close(fd);
      auto path = *iter + name;
      if (next) {
        if (path != curPath)
          continue;
        else
          next = false;
      } else {
        if (path == curPath)
          continue;
        if (libHeader && !next)
          searchPaths_.pop_back();
        else
          searchPaths_.pop_front();
        return new std::string(path);
      }
    } else if (errno == EMFILE) {
      Error("may recursive include");
    }
  }
  return nullptr;
}


void Preprocessor::AddMacro(const std::string& name,
                            std::string* text,
                            bool preDef) {
  TokenSequence ts;
  Scanner scanner(text);
  scanner.Tokenize(ts);
  Macro macro(ts, preDef);

  AddMacro(name, macro);
}


static std::string* Date() {
  time_t t = time(NULL);
  struct tm* tm = localtime(&t);
  char buf[14];
  strftime(buf, sizeof buf, "\"%a %M %Y\"", tm);
  return new std::string(buf);
}


void Preprocessor::Init() {
  // Preinclude search paths
  AddSearchPath("/usr/local/include/");
  AddSearchPath("/usr/include/x86_64-linux-gnu/");
  AddSearchPath("/usr/include/linux/");
  AddSearchPath("/usr/include/");
  AddSearchPath("/usr/local/wgtcc/include/");

  // The __FILE__ and __LINE__ macro is empty
  // They are handled seperately
  AddMacro("__FILE__", Macro(TokenSequence(), true));
  AddMacro("__LINE__", Macro(TokenSequence(), true));

  AddMacro("__DATE__", Date(), true);
  AddMacro("__STDC__", new std::string("1"), true);
  AddMacro("__STDC__HOSTED__", new std::string("0"), true);
  AddMacro("__STDC_VERSION__", new std::string("201103L"), true);
}


void Preprocessor::HandleTheFileMacro(TokenSequence& os, const Token* macro) {
  auto file = Token::New(*macro);
  file->tag_ = Token::LITERAL;
  file->str_ = "\"" + *macro->loc_.filename_ + "\"";
  os.InsertBack(file);
}


void Preprocessor::HandleTheLineMacro(TokenSequence& os, const Token* macro) {
  auto line = Token::New(*macro);
  line->tag_ = Token::I_CONSTANT;
  line->str_ = std::to_string(macro->loc_.line_);
  os.InsertBack(line);
}


void Preprocessor::UpdateFirstTokenLine(TokenSequence ts) {
  auto loc = ts.Peek()->loc_;
  loc.line_ = curLine_  + loc.line_ - lineLine_ - 1;
  ts.UpdateHeadLocation(loc);
}


TokenSequence Macro::RepSeq(const std::string* filename, unsigned line) {
  // Update line
  TokenList tl;
  TokenSequence ret(&tl);
  ret.Copy(repSeq_);
  auto ts = ret;
  while (!ts.Empty()) {
    auto loc = ts.Peek()->loc_;
    loc.filename_ = filename;
    loc.line_ = line;
    ts.UpdateHeadLocation(loc);
    ts.Next();
  }
  return ret;
}


void Preprocessor::AddSearchPath(std::string path) {
  if (path.back() != '/')
    path += "/";
  if (path[0] != '/')
    path = "./" + path;
  searchPaths_.push_front(path);
}
