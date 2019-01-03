#include "ir/basic_block.h"
#include "ir/instructions.h"
#include "ir/type.h"

namespace tdl {
namespace ir {

class phi_node;

basic_block::basic_block(context &ctx, const std::string &name, function *parent):
  value(type::get_label_ty(ctx), name), ctx_(ctx), parent_(parent){

}

basic_block* basic_block::create(context &ctx, const std::string &name, function *parent){
  return new basic_block(ctx, name, parent);
}

basic_block::iterator basic_block::get_first_non_phi(){
  auto it = begin();
  for(; it != end(); it++)
  if(!dynamic_cast<phi_node*>(*it))
    return it;
  return it;
}

}

}
