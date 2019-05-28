#ifndef TRITON_INCLUDE_LANG_MODULE_H
#define TRITON_INCLUDE_LANG_MODULE_H

#include "node.h"

namespace triton{
namespace lang{

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
