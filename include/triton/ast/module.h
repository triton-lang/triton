#ifndef TRITON_INCLUDE_AST_MODULE_H
#define TRITON_INCLUDE_AST_MODULE_H

#include "ops.h"
#include "parser.hpp"
#include "node.h"
#include <cassert>
#include <vector>
#include <string>
#include <iostream>


namespace triton{
namespace ast{

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

}

}

#endif
