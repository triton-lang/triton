#ifndef TDL_INCLUDE_IR_CONTEXT_H
#define TDL_INCLUDE_IR_CONTEXT_H

namespace tdl{
namespace ir{

class type;

/* Context */
class context {
public:
  type *get_void_ty();
  type *get_int1_ty();

private:
};

}
}

#endif
