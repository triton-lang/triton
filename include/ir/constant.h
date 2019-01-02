#ifndef TDL_INCLUDE_IR_CONSTANT_H
#define TDL_INCLUDE_IR_CONSTANT_H

#include "value.h"

namespace tdl{
namespace ir{

class type;
class context;

/* Constant */
class constant: public value{
public:
  static constant* get_all_ones_value(type *ty);
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

/* Constant int */
class constant_int: public constant{

};

/* constant fp */
class constant_fp: public constant{
public:
  static constant* get_zero_value_for_negation(type *ty);
};


}
}

#endif
