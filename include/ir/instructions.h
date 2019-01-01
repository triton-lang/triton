#ifndef TDL_INCLUDE_IR_INSTRUCTIONS_H
#define TDL_INCLUDE_IR_INSTRUCTIONS_H

#include "value.h"

namespace tdl{
namespace ir{

/* Instructions */
class instruction: public value{

};

class phi_node: public instruction{

};

class binary_operator: public instruction{

};

class unary_operator: public instruction{

};

}
}

#endif
