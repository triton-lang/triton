#ifndef TDL_INCLUDE_IR_TYPE_H
#define TDL_INCLUDE_IR_TYPE_H

#include <vector>

namespace tdl{
namespace ir{

class context;
class value;

/* Type */
class type {
public:
  virtual ~type(){}

  // accessors
  context &get_context() const;

  // type attributes
  unsigned get_fp_mantissa_width() const;
  unsigned get_integer_bit_width() const;
  unsigned get_scalar_bitsize() const;
  const std::vector<unsigned> &get_tile_shapes() const;
  type *get_scalar_ty() const;
  unsigned get_pointer_address_space() const;

  // type predicates
  bool is_int_or_tileint_ty();
  bool is_integer_ty() const;
  bool is_integer_ty(unsigned width) const;
  bool is_pointer_ty() const;
  bool is_float_ty() const;
  bool is_double_ty() const;
  bool is_floating_point_ty() const;
  bool is_sized() const;
  bool is_tile_ty() const;

  // Factory methods
  static type* get_void_ty(context &ctx);
  static type* get_float_ty(context &ctx);
  static type* get_double_ty(context &ctx);

};

class integer_type: public type {
public:
  static integer_type* get(context &ctx, unsigned width);
};

class composite_type: public type{
public:
  bool index_valid(value *idx) const;
  type* get_type_at_index(value *idx) const;
};

class tile_type: public type {
public:
  static tile_type* get(type *ty, const std::vector<unsigned> &shapes);
  static tile_type* get_same_shapes(type *ty, type *ref);
};

class pointer_type: public type {
public:
  static pointer_type* get(type *ty, unsigned address_space);
  type *get_element_ty() const;
};

class function_type: public type {
public:
  static function_type* get(type *ret_ty, const std::vector<type*>& param_tys);

private:
  type *return_type_;
  std::vector<type *> param_types_;
};


}
}

#endif
