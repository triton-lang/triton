#ifndef TRITON_INCLUDE_AST_ERROR_H
#define TRITON_INCLUDE_AST_ERROR_H

#include "ops.h"
#include "parser.hpp"
#include "node.h"
#include <cassert>
#include <vector>
#include <string>
#include <iostream>


namespace triton{


namespace ir{
  class function;
  class value;
  class type;
  class builder;
  class module;
}

namespace ast{

class expression;
class pointer;
class identifier;
class constant;
class compound_statement;
class initializer;
class declaration_specifier;
class function;

/* Translation Unit */
class translation_unit: public node{
public:
  translation_unit(node *item)
    : decls_(item) { }

  translation_unit *add(node *item) {
    decls_.append(item);
    return this;
  }

  ir::value* codegen(ir::module * mod) const;

private:
  list<node*> decls_;
};

void update_location(const char *t);
void print_error(const char *error);
char return_impl(char t, const char * yytext);
yytokentype return_impl(yytokentype t, const char * yytext);
void return_void(const char * yytext);

}

}

#endif
