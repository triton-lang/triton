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
  uint64_t get_value() const { return value_; }
  static constant *get(type *ty, uint64_t value);

private:
  uint64_t value_;
};

/* constant fp */
class constant_fp: public constant{
  constant_fp(context &ctx, double value);

public:
  double get_value() { return value_; }
  static constant* get_negative_zero(type *ty);
  static constant* get_zero_value_for_negation(type *ty);
  static constant *get(context &ctx, double v);

private:
  double value_;
};

/* global value */
class global_value: public constant {
public:
  enum linkage_types_t {
    external
  };

public:
  global_value(type *ty, unsigned num_ops,
               linkage_types_t linkage, const std::string &name,
               unsigned addr_space);

private:
  linkage_types_t linkage_;
};

/* global object */
class global_object: public global_value {
public:
  global_object(type *ty, unsigned num_ops,
               linkage_types_t linkage, const std::string &name,
               unsigned addr_space = 0);
};


}
}

#endif
