#ifndef TDL_INCLUDE_IR_TYPE_H
#define TDL_INCLUDE_IR_TYPE_H

#include <vector>

namespace tdl{
namespace ir{

class context;

/* Type */
class type {
public:
  bool is_integer_ty() const;
  bool is_pointer_ty() const;
  bool is_float_ty() const;
  bool is_double_ty() const;
  bool is_floating_point_ty() const;

  // type attributes
  unsigned get_fp_mantissa_width() const;
  unsigned get_integer_bit_width() const;
  const std::vector<unsigned> &get_tile_shapes() const;
  // Factory methods
  static type* get_void_ty(context &ctx);
  static type* get_float_ty(context &ctx);
  static type* get_double_ty(context &ctx);
};

class integer_type: public type {
public:
  static integer_type* get(context &ctx, unsigned width);
};

class tile_type: public type {
public:
  static tile_type* get(type *ty, const std::vector<unsigned> &shapes);
};

class pointer_type: public type {
public:
  static pointer_type* get(type *ty, unsigned address_space);
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
