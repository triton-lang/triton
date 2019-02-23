#ifndef TDL_INCLUDE_IR_CONSTANT_H
#define TDL_INCLUDE_IR_CONSTANT_H

#include "value.h"
#include <cassert>

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
protected:
  constant_int(type *ty, uint64_t value);

public:
  uint64_t get_value() const { return value_; }
  static constant_int *get(type *ty, uint64_t value);

protected:
  uint64_t value_;
};

/* Metaparameter int */
class metaparameter: public constant_int{
  metaparameter(type *ty, unsigned lo, unsigned hi);

public:
  static metaparameter *create(context &ctx, type *ty, unsigned lo, unsigned hi);
  void set_value(uint64_t value) { value_ = value; }

private:
  unsigned lo_;
  unsigned hi_;
};

/* constant range */
class constant_range: public constant{
  constant_range(type *ty, constant_int* first, constant_int* last);

public:
  static constant *get(constant_int *first, constant_int *last);

private:
  constant_int* first_;
  constant_int* last_;
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
