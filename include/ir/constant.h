#ifndef TDL_INCLUDE_IR_CONSTANT_H
#define TDL_INCLUDE_IR_CONSTANT_H

#include "value.h"

namespace tdl{
namespace ir{

class type;
class context;

/* Constant */
class constant: public user{
protected:
  using user::user;

public:
  static constant* get_all_ones_value(type *ty);
  static constant* get_null_value(type *ty);
};

/* Undef value */
class undef_value: public constant{
private:
  undef_value(type *ty);

public:
  static undef_value* get(type* ty);
};

/* Constant int */
class constant_int: public constant{
  constant_int(type *ty, uint64_t value);

public:
  static constant *get(type *ty, uint64_t value);

private:
  uint64_t value_;
};

/* constant fp */
class constant_fp: public constant{
  constant_fp(context &ctx, double value);

public:
  static constant* get_negative_zero(type *ty);
  static constant* get_zero_value_for_negation(type *ty);
  static constant *get(context &ctx, double v);

private:
  double value_;
};


}
}

#endif
