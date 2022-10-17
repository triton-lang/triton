#pragma once

#ifndef _TRITON_IR_TYPE_H_
#define _TRITON_IR_TYPE_H_

#include <cassert>
#include <vector>
#include <string>
#include <stdexcept>

namespace triton{
namespace ir{

class context;
class value;
class integer_type;
class constant_int;

/* Type */
class type {
public:
  typedef std::vector<unsigned>	         block_shapes_t;

  typedef std::vector<type*>                  contained_tys_vec_t;
  typedef contained_tys_vec_t::iterator       ty_iterator;
  typedef contained_tys_vec_t::const_iterator const_ty_iterator;

public:
  enum id_t {
    // primitive types
    VoidTyID = 0,    ///< type with no size
    FP8TyID,         ///< 8-bit floating point type (3 bits mantissa)
    FP16TyID,        ///< 16-bit floating point type (10 bits mantissa)
    BF16TyID,        ///< 16-bit floating point type (7 bits mantissa)
    FP32TyID,        ///< 32-bit floating point type
    FP64TyID,        ///< 64-bit floating point type
    LabelTyID,       ///< Labels
    MetadataTyID,    ///< Metadata
    TokenTyID,       ///< Token
    // derived types
    IntegerTyID,     ///< Arbitrary bit width integers
    FunctionTyID,    ///< Functions
    PointerTyID,     ///< Pointers
    StructTyID,      ///< Struct
    BlockTyID,       ///< Block
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
  block_shapes_t get_block_shapes() const;
  const size_t get_tile_rank() const;
  const size_t get_tile_ranks1() const;
  unsigned get_tile_num_elements() const;
  type *get_tile_element_ty() const;
  unsigned get_pointer_address_space() const;
  type *get_pointer_element_ty() const;
  unsigned get_struct_numel() const { return contained_tys_.size(); }
  type *get_struct_type(unsigned int i) const { return contained_tys_[i]; }

  // primitive predicates
  bool is_void_ty() const               { return id_ == VoidTyID; }
  bool is_fp8_ty() const                { return id_ == FP8TyID; }
  bool is_fp16_ty() const               { return id_ == FP16TyID; }
  bool is_bf16_ty() const               { return id_ == BF16TyID; }
  bool is_fp32_ty() const               { return id_ == FP32TyID; }
  bool is_fp64_ty() const               { return id_ == FP64TyID; }
  bool is_label_ty()  const             { return id_ == LabelTyID;}
  bool is_metadata_ty() const           { return id_ == MetadataTyID; }
  bool is_token_ty() const              { return id_ == TokenTyID; }
  bool is_integer_ty() const            { return id_ == IntegerTyID; }
  bool is_bool_ty() const               { return is_integer_ty(1); }
  bool is_pointer_ty() const            { return id_ == PointerTyID; }
  bool is_block_ty() const               { return id_ == BlockTyID; }
  bool is_struct_ty() const             { return id_ == StructTyID; }

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
  static type *get_fp8_ty(context &ctx);
  static type *get_fp16_ty(context &ctx);
  static type *get_bf16_ty(context &ctx);
  static type *get_fp32_ty(context &ctx);
  static type *get_fp64_ty(context &ctx);
  // integer types
  static integer_type *get_int1_ty(context &ctx);
  static integer_type *get_int8_ty(context &ctx);
  static integer_type *get_int16_ty(context &ctx);
  static integer_type *get_int32_ty(context &ctx);
  static integer_type *get_int64_ty(context &ctx);
  static integer_type *get_int128_ty(context &ctx);

  // repr
  std::string tile_repr() const {
    std::string res = get_tile_element_ty()->repr();
    auto shapes = get_block_shapes();
    res += "<";
    for(size_t i = 0; i < shapes.size(); i++){
      if(i > 0)
        res += ", ";
      res += std::to_string(shapes[i]);
    }
    res+= ">";
    return res;
  }

  std::string repr() const {
    switch(id_) {
      case VoidTyID: return "void";
      case FP8TyID: return "fp8";
      case BF16TyID: return "bf16";
      case FP16TyID: return "f16";
      case FP32TyID: return "f32";
      case FP64TyID: return "f64";
      case LabelTyID: return "label";
      case MetadataTyID: return "md";
      case TokenTyID: return "tok";
      case IntegerTyID: return ("i") + std::to_string(get_integer_bitwidth());
      case FunctionTyID: return "fn";
      case PointerTyID: return get_pointer_element_ty()->repr() + "*";
      case StructTyID: return "struct";
      case BlockTyID: return tile_repr();
      default: break;
    }
    throw std::logic_error("unknown type id '" + std::to_string(id_) + "'");
  };

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
    : type(ctx, IntegerTyID), bitwidth_(bitwidth) {}

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

class struct_type: public composite_type {
public:
  struct_type(const contained_tys_vec_t& tys, bool is_packed);
  unsigned get_num_types() const { return contained_tys_.size(); }
  static struct_type* get(const contained_tys_vec_t& tys, bool is_packed);

private:
  bool is_packed_;
};

class block_type: public composite_type {
private:
  block_type(type *ty, const block_shapes_t &shapes);
  static bool is_valid_elt_ty(type *ty);

public:
  // accessors
  const block_shapes_t& get_shapes() const { return shapes_; }
  unsigned get_num_elements() const;
  unsigned get_bitwidth() const;

  // factory methods
  static block_type* get(type *ty, const block_shapes_t &shapes);
  static block_type* get_same_shapes(type *ty, type *ref);

private:
  block_shapes_t shapes_;
};

class pointer_type: public type {
private:
  pointer_type(type *ty, unsigned address_space);
  static bool is_valid_elt_ty(type *ty);

public:
  // accessors
  unsigned get_address_space()               const { return address_space_; }
  type *get_element_ty()                     const { return contained_tys_[0]; }
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
  void     reset_ret_ty(type* ty)         { contained_tys_[0] = ty;}
  // factory methods
  static function_type* get(type *ret_ty, const std::vector<type*>& param_tys);
};


}
}

#endif
