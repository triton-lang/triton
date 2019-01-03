#ifndef TDL_INCLUDE_IR_TYPE_H
#define TDL_INCLUDE_IR_TYPE_H

#include <vector>

namespace tdl{
namespace ir{

class context;
class value;
class integer_type;

/* Type */
class type {
public:
  enum id_t {
    // primitive types
    VoidTyID = 0,    ///<  0: type with no size
    HalfTyID,        ///<  1: 16-bit floating point type
    FloatTyID,       ///<  2: 32-bit floating point type
    DoubleTyID,      ///<  3: 64-bit floating point type
    LabelTyID,       ///<  4: Labels
    MetadataTyID,    ///<  5: Metadata
    TokenTyID,       ///<  6: Token
    // derived types
    IntegerTyID,     ///< 7: Arbitrary bit width integers
    FunctionTyID,    ///< 8: Functions
    PointerTyID,     ///< 9: Pointers
    TileTyID,        ///< 10: Tile
  };

public:
  //constructors
  type(context &ctx, id_t id) : ctx_(ctx), id_(id) {}

  //destructor
  virtual ~type(){}

  // accessors
  context &get_context() const { return ctx_; }

  // type attributes
  unsigned get_fp_mantissa_width() const;
  unsigned get_integer_bitwidth() const;
  type *get_scalar_ty() const;
  const std::vector<unsigned> &get_tile_shapes() const;
  type *get_tile_element_ty() const;
  unsigned get_pointer_address_space() const;

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
  std::vector<type*> contained_tys_;
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
  tile_type(type *ty, const std::vector<unsigned> &shapes);
  static bool is_valid_elt_ty(type *ty);

public:
  // accessors
  const std::vector<unsigned>& get_shapes() const { return shapes_; }

  // factory methods
  static tile_type* get(type *ty, const std::vector<unsigned> &shapes);
  static tile_type* get_same_shapes(type *ty, type *ref);

private:
  std::vector<unsigned> shapes_;
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
public:
  static function_type* get(type *ret_ty, const std::vector<type*>& param_tys);

private:
  type *return_type_;
  std::vector<type *> param_types_;
};


}
}

#endif
