#ifndef _WGTCC_SCOPE_H_
#define _WGTCC_SCOPE_H_

#include <iostream>
#include <map>
#include <string>
#include <vector>


class Identifier;
class Token;


enum ScopeType {
  S_FILE,
  S_PROTO,
  S_BLOCK,
  S_FUNC,
};


class Scope {
  friend class StructType;
  using TagList = std::vector<Identifier*>;
  using IdentMap = std::map<std::string, Identifier*>;

public:
  explicit Scope(Scope* parent, enum ScopeType type)
      : parent_(parent), type_(type) {}
  ~Scope() {}
  Scope* Parent() { return parent_; }
  void SetParent(Scope* parent) { parent_ = parent; }
  enum ScopeType Type() const { return type_; }

  Identifier* Find(const Token* tok);
  Identifier* FindInCurScope(const Token* tok);
  Identifier* FindTag(const Token* tok);
  Identifier* FindTagInCurScope(const Token* tok);
  TagList AllTagsInCurScope() const;

  void Insert(Identifier* ident);
  void Insert(const std::string& name, Identifier* ident);
  void InsertTag(Identifier* ident);
  void Print();
  bool operator==(const Scope& other) const { return type_ == other.type_; }
  IdentMap::iterator begin() { return identMap_.begin(); }
  IdentMap::iterator end() { return identMap_.end(); }
  size_t size() const { return identMap_.size(); }

private:
  Identifier* Find(const std::string& name);
  Identifier* FindInCurScope(const std::string& name);
  Identifier* FindTag(const std::string& name);
  Identifier* FindTagInCurScope(const std::string& name);
  std::string TagName(const std::string& name) {
    return name + "@:tag";
  }
  static bool IsTagName(const std::string& name) {
    return name.size() > 5 && name[name.size() - 5] == '@';
  }
  const Scope& operator=(const Scope& other);
  Scope(const Scope& scope);

  Scope* parent_;
  enum ScopeType type_;

  IdentMap identMap_;
};

#endif
