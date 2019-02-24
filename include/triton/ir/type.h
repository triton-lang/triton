#ifndef TDL_INCLUDE_IR_TYPE_H
#define TDL_INCLUDE_IR_TYPE_H

#include <vector>
#include <iostream>
#include <set>

namespace tdl{
namespace ir{

class context;
class value;
class integer_type;
class constant_int;

/* Type */
class type {
public:
  typedef std::vector<constant_int*>	         tile_shapes_t;

protected:
  typedef std::vector<type*>                  contained_tys_vec_t;
  typedef contained_tys_vec_t::iterator       ty_iterator;
  typedef contained_tys_vec_t::const_iterator const_ty_iterator;

public:
  enum id_t {
    // primitive types
    VoidTyID = 0,    ///<  0: type with no size
    HalfTyID,        ///<  1: 16-bit floating point type
    FloatTyID,       ///<  2: 32-bit floating point type
    DoubleTyID,      ///<  3: 64-bit floating point type
    X86_FP80TyID,    ///<  4: 80-bit floating point type (X87)
    FP128TyID,       ///<  5: 128-bit floating point type (112-bit mantissa)
    PPC_FP128TyID,   ///<  6: 128-bit floating point type (two 64-bits, PowerPC)
    LabelTyID,       ///<  7: Labels
    MetadataTyID,    ///<  8: Metadata
    TokenTyID,       ///<  9: Token
    // derived types
    IntegerTyID,     ///< 10: Arbitrary bit width integers
    FunctionTyID,    ///< 11: Functions
    PointerTyID,     ///< 12: Pointers
    TileTyID,        ///< 13: Tile
  };

public:
  //constructors
  type(context &ctx, id_t id) : ctx_(ctx), id_(id) { }

  //destructor
  virtual ~type(){}

  // accessors
  context &get_context() const { return ctx_; }
  id_t get_type_id() const     { return id_;  }
  // type attributes
  unsigned get_fp_mantissa_width() const;
  unsigned get_integer_bitwidth() const;
  unsigned get_tile_bitwidth() const;
  unsigned get_primitive_size_in_bits() const;
  type *get_scalar_ty() const;
  const tile_shapes_t& get_tile_shapes() const;
  unsigned get_tile_num_elements() const;
  type *get_tile_element_ty() const;
  unsigned get_pointer_address_space() const;
  type *get_pointer_element_ty() const;

  // primitive predicates
  bool is_void_ty() const     { return id_ == VoidTyID; }
  bool is_half_ty() const     { return id_ == HalfTyID; }
  bool is_float_ty() const    { return id_ == FloatTyID; }
  bool is_double_ty() const   { return id_ == DoubleTyID; }
  bool is_label_ty()  const   { return id_ == LabelTyID;}
  bool is_metadata_ty() const { return id_ == MetadataTyID; }
  bool is_token_ty() const    { return id_ == TokenTyID; }
  bool is_integer_ty() const  { return id_ == IntegerTyID; }
  bool is_pointer_ty() const  { return id_ == PointerTyID; }
  bool is_tile_ty() const     { return id_ == TileTyID; }

  // Composite predicates
  bool is_int_or_tileint_ty();
  bool is_integer_ty(unsigned width) const;
  bool is_floating_point_ty() const;
  bool is_sized() const ;

  // Factory methods
  // primitive types
  static type *get_void_ty(context &ctx);
  static type *get_label_ty(context &ctx);
  // half
  static type *get_half_ty(context &ctx);
  static type *get_float_ty(context &ctx);
  static type *get_double_ty(context &ctx);
  // integer types
  static integer_type *get_int1_ty(context &ctx);
  static integer_type *get_int8_ty(context &ctx);
  static integer_type *get_int16_ty(context &ctx);
  static integer_type *get_int32_ty(context &ctx);
  static integer_type *get_int64_ty(context &ctx);
  static integer_type *get_int128_ty(context &ctx);


private:
  context &ctx_;
  id_t id_;

protected:
  contained_tys_vec_t contained_tys_;
};

class integer_type: public type {
  friend class context_impl;

private:
  // constructors
  integer_type(context &ctx, unsigned bitwidth)
    : type(ctx, IntegerTyID), bitwidth_(bitwidth){ }

public:
  // accessors
  unsigned get_bitwidth() const { return bitwidth_; }

  // factory methods
  static integer_type* get(context &ctx, unsigned width);

private:
  unsigned bitwidth_;
};

class composite_type: public type{
protected:
  using type::type;

public:
  bool index_valid(value *idx) const;
  type* get_type_at_index(value *idx) const;
};

class tile_type: public composite_type {
private:
  tile_type(type *ty, const tile_shapes_t &shapes);
  static bool is_valid_elt_ty(type *ty);

public:
  // accessors
  const tile_shapes_t& get_shapes() const { return shapes_; }
  unsigned get_num_elements() const;
  unsigned get_bitwidth() const;

  // factory methods
  static tile_type* get(type *ty, const tile_shapes_t &shapes);
  static tile_type* get_same_shapes(type *ty, type *ref);

  // shortcut to get a 1 element in the shape
  static tile_shapes_t::value_type make_one(context &ctx);

private:
  tile_shapes_t shapes_;
};

class pointer_type: public type {
private:
  pointer_type(type *ty, unsigned address_space);
  static bool is_valid_elt_ty(type *ty);

public:
  // accessors
  unsigned get_address_space() const { return address_space_; }
  type *get_element_ty()       const { return contained_tys_[0]; }

  // factory methods
  static pointer_type* get(type *ty, unsigned address_space);

private:
  unsigned address_space_;
};

class function_type: public type {
private:
  function_type(type *ret_ty, const std::vector<type *> &param_tys);

public:
  // accessors
  unsigned get_num_params()         const { return contained_tys_.size() - 1;  }
  const_ty_iterator params_begin() const { return contained_tys_.begin() + 1; }
  const_ty_iterator params_end()   const { return contained_tys_.end(); }
  ty_iterator       params_begin()       { return contained_tys_.begin() + 1; }
  ty_iterator       params_end()         { return contained_tys_.end(); }
  type*    get_param_ty(unsigned i) const { return contained_tys_.at(1 + i);   }
  type*    get_return_ty()          const { return contained_tys_.at(0);       }
  // factory methods
  static function_type* get(type *ret_ty, const std::vector<type*>& param_tys);
};


}
}

#endif
