#ifndef TDL_INCLUDE_IR_CONSTANT_H
#define TDL_INCLUDE_IR_CONSTANT_H

#include "value.h"

namespace tdl{
namespace ir{

class type;
class context;

/* Constant */
class constant: public value{

};

/* Undef value */
class undef_value: public constant{
public:
  static undef_value* get(type* ty);
};

/* Data array */
class constant_data_array: public constant{
public:
  static constant_data_array* get_string(context &ctx, const std::string &str);
};

}
}

#endif
