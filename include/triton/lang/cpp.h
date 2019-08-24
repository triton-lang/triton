#ifndef _WGTCC_CPP_H_
#define _WGTCC_CPP_H_

#include "scanner.h"

#include <cstdio>
#include <list>
#include <map>
#include <set>
#include <stack>
#include <string>

class Macro;
struct CondDirective;

using MacroMap = std::map<std::string, Macro>;
using ParamList = std::list<std::string>;
using ParamMap = std::map<std::string, TokenSequence>;
using PPCondStack = std::stack<CondDirective>;
using PathList = std::list<std::string>;


class Macro {
public:
  Macro(const TokenSequence& repSeq, bool preDef=false)
      : funcLike_(false), variadic_(false),
        preDef_(preDef), repSeq_(repSeq) {}

  Macro(bool variadic, ParamList& params,
        TokenSequence& repSeq, bool preDef=false)
      : funcLike_(true), variadic_(variadic), preDef_(preDef),
        params_(params), repSeq_(repSeq) {}

  ~Macro() {}
  bool FuncLike() { return funcLike_; }
  bool ObjLike() { return !FuncLike(); }
  bool Variadic() { return variadic_; }
  bool PreDef() { return preDef_; }
  ParamList& Params() { return params_; }
  TokenSequence RepSeq(const std::string* filename, unsigned line);

private:
  bool funcLike_;
  bool variadic_;
  bool preDef_;
  ParamList params_;
  TokenSequence repSeq_;
};


struct CondDirective {
  int tag_;
  bool enabled_;
  bool cond_;
};


class Preprocessor {
public:
  Preprocessor(const std::string* str, bool isSrc = true)
      : curLine_(1), lineLine_(0), curCond_(true), fName_(nullptr), fSrc_(nullptr) {
    if(isSrc)
      fSrc_ = str;
    else
      fName_ = str;
    // Add predefined
    Init();
  }


  ~Preprocessor() {}
  void Finalize(TokenSequence os);
  void Process(TokenSequence& os);
  void Expand(TokenSequence& os, TokenSequence is, bool inCond=false);
  void Subst(TokenSequence& os, TokenSequence is,
             bool leadingWS, const HideSet& hs, ParamMap& params);
  void Glue(TokenSequence& os, TokenSequence is);
  void Glue(TokenSequence& os, const Token* tok);
  const Token* Stringize(TokenSequence is);
  void Stringize(std::string& str, TokenSequence is);
  const Token* ParseActualParam(TokenSequence& is, Macro* macro, ParamMap& paramMap);
  int GetDirective(TokenSequence& is);
  const Token* EvalDefOp(TokenSequence& is);
  void ReplaceIdent(TokenSequence& is);
  void ParseDirective(TokenSequence& os, TokenSequence& is, int directive);
  void ParseIf(TokenSequence ls);
  void ParseIfdef(TokenSequence ls);
  void ParseIfndef(TokenSequence ls);
  void ParseElif(TokenSequence ls);
  void ParseElse(TokenSequence ls);
  void ParseEndif(TokenSequence ls);
  void ParseInclude(TokenSequence& is, TokenSequence ls);
  void ParseDef(TokenSequence ls);
  void ParseUndef(TokenSequence ls);
  void ParseLine(TokenSequence ls);
  void ParseError(TokenSequence ls);
  void ParsePragma(TokenSequence ls);
  void IncludeSrc(TokenSequence& is, const std::string* text, const std::string* filename);
  void IncludeFile(TokenSequence& is, const std::string* filename);
  bool ParseIdentList(ParamList& params, TokenSequence& is);


  Macro* FindMacro(const std::string& name) {
    auto res = macroMap_.find(name);
    if (res == macroMap_.end())
      return nullptr;
    return &res->second;
  }

  void AddMacro(const std::string& name,
                std::string* text, bool preDef=false);

  void AddMacro(const std::string& name, const Macro& macro) {
    auto res = macroMap_.find(name);
    if (res != macroMap_.end()) {
      // TODO(wgtdkp): give warning
      macroMap_.erase(res);
    }
    macroMap_.insert(std::make_pair(name, macro));
  }

  void RemoveMacro(const std::string& name) {
    auto res = macroMap_.find(name);
    if (res == macroMap_.end())
      return;
    if(res->second.PreDef()) // Cannot undef predefined macro
      return;
    macroMap_.erase(res);
  }

  std::string* SearchFile(const std::string& name,
                          const bool libHeader,
                          bool next,
                          const std::string& curPath);

  void AddSearchPath(std::string path);
  void HandleTheFileMacro(TokenSequence& os, const Token* macro);
  void HandleTheLineMacro(TokenSequence& os, const Token* macro);
  void UpdateFirstTokenLine(TokenSequence ts);

  bool NeedExpand() const {
    if (ppCondStack_.empty())
      return true;
    auto top = ppCondStack_.top();
    return top.enabled_ && top.cond_;
  }

private:
  void Init();

  PPCondStack ppCondStack_;
  unsigned curLine_;
  unsigned lineLine_;
  bool curCond_;

  MacroMap macroMap_;
  PathList searchPaths_;
  const std::string* fName_;
  const std::string* fSrc_;
};

#endif
