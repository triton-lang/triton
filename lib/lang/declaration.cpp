#include "triton/lang/statement.h"
#include "triton/lang/declaration.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/builder.h"
#include "triton/ir/type.h"
#include "triton/ir/metadata.h"


namespace triton{

namespace lang{

/* Declaration specifier */
ir::type* typed_declaration_specifier::type(ir::module *mod) const {
  ir::context &ctx = mod->get_context();
  switch (ty_) {
  case VOID_T:      return ir::type::get_void_ty(ctx);
  case INT1_T:      return ir::type::get_int1_ty(ctx);
  case INT8_T:      return ir::type::get_int8_ty(ctx);
  case INT16_T:     return ir::type::get_int16_ty(ctx);
  case INT32_T:     return ir::type::get_int32_ty(ctx);
  case INT64_T:     return ir::type::get_int64_ty(ctx);
  case FLOAT16_T:   return ir::type::get_half_ty(ctx);
  case FLOAT32_T:   return ir::type::get_float_ty(ctx);
  case FLOAT64_T:   return ir::type::get_double_ty(ctx);
  default:          throw std::runtime_error("unreachable");
  }
}

std::vector<modifier*> typed_declaration_specifier::modifiers() const {
  return {};
}


ir::type* declaration_modifier::type(ir::module *mod) const {
  return decl_spec_->type(mod);
}

std::vector<modifier*> declaration_modifier::modifiers() const {
  auto result = decl_spec_->modifiers();
  result.push_back(mod_);
  return result;
}


/* Parameter */
ir::type* parameter::type(ir::module *mod) const {
  return decl_->type(mod, spec_->type(mod), {});
}

std::vector<modifier*> parameter::modifiers() const {
  return spec_->modifiers();
}

const identifier *parameter::id() const {
  return decl_->id();
}

/* Declarators */
ir::type* declarator::type(ir::module *mod, ir::type *type, storage_spec_vec_const_ref_t storage) const{
  if(ptr_)
    return type_impl(mod, ptr_->type(mod, type, storage), storage);
  return type_impl(mod, type, storage);
}

// Identifier
ir::type* identifier::type_impl(ir::module *, ir::type *type, storage_spec_vec_const_ref_t) const{
  return type;
}

const std::string &identifier::name() const{
  return name_;
}

// Tile
ir::type* tile::type_impl(ir::module *mod, ir::type *type, storage_spec_vec_const_ref_t) const{
  ir::type::tile_shapes_t shapes;
  for(expression *expr: shapes_->values()){
    ir::constant_int *shape = dynamic_cast<ir::constant_int*>(expr->codegen(mod));
    assert(shape);
    shapes.push_back(shape);
  }
  return ir::tile_type::get(type, shapes);
}


// Pointer
ir::type* pointer::type_impl(ir::module*, ir::type *type, storage_spec_vec_const_ref_t storage) const{
  auto is_cst = [](modifier* x){ return x->is_cst_space(); };
  bool is_ptr_to_const = std::find_if(storage.begin(), storage.end(), is_cst) != storage.end();
  return ir::pointer_type::get(type, is_ptr_to_const?4:1);
}

// Function
void function::bind_parameters(ir::module *mod, ir::function *fn) const{
  std::vector<ir::argument*> args = fn->args();
  assert(args.size() == args_->values().size());
  for(size_t i = 0; i < args.size(); i++){
    parameter *param_i = args_->values().at(i);
    const identifier *id_i = param_i->id();
    if(id_i){
      args[i]->set_name(id_i->name());
      mod->set_value(id_i->name(), nullptr, args[i]);
      mod->get_scope().types[id_i->name()] = args[i]->get_type();
    }
  }
}

ir::type* function::type_impl(ir::module* mod, ir::type *type, storage_spec_vec_const_ref_t) const{
  std::vector<ir::type*> types;
  for(parameter* param: args_->values())
    types.push_back(param->type(mod));
  return ir::function_type::get(type, types);
}


/* Declaration */
ir::value* declaration::codegen(ir::module* mod) const{
  for(initializer *init: init_->values())
    init->set_specifier(spec_);
  init_->codegen(mod);
  return nullptr;
}

/* Initializer */
ir::type* initializer::type_impl(ir::module *mod, ir::type *type, storage_spec_vec_const_ref_t storage) const{
  return decl_->type(mod, type, storage);
}

void initializer::set_specifier(const declaration_specifier *spec) {
  spec_ = spec;
}

ir::value* initializer::codegen(ir::module * mod) const{
  std::vector<modifier*> modifiers = spec_->modifiers();
  ir::type *ty = decl_->type(mod, spec_->type(mod), modifiers);
  std::string name = decl_->id()->name();
  ir::value *value = ir::undef_value::get(ty);
  auto is_tunable = [](modifier* x){ return x->is_tunable(); };
  if(std::find_if(modifiers.begin(), modifiers.end(), is_tunable) != modifiers.end()){
    auto csts = dynamic_cast<list<constant*>*>((node*)expr_);
    if(csts == nullptr)
      throw std::runtime_error("must specify constant list for metaparameters");
    std::vector<unsigned> values;
    for(constant* cst: csts->values())
      values.push_back(cst->value());
    value = ir::metaparameter::create(mod->get_context(), ty, values);
    mod->register_global(name, value);
  }
  else if(expr_){
    value = expr_->codegen(mod);
    value = explicit_cast(mod->get_builder(), value, ty);
    implicit_broadcast(mod, ty, value);
  }
  value->set_name(name);
  // metadata
  auto is_multiple_of = [](modifier* x){ return x->is_multiple_of(); };
  auto it = std::find_if(modifiers.begin(), modifiers.end(), is_multiple_of);
  if(it != modifiers.end())
    (*it)->add_metadata(mod, name);
  // register
  mod->set_value(name, value);
  mod->get_scope().types[name] = ty;
  if(auto *x = dynamic_cast<ir::alloc_const*>(value))
    mod->add_alloc(x);
  // constants
  auto is_cst = [](modifier* x){ return x->is_cst(); };
  if(std::find_if(modifiers.begin(), modifiers.end(), is_cst) != modifiers.end())
    mod->set_const(name);
  return value;
}

/* Type name */
ir::type *type_name::type(ir::module *mod) const{
  return decl_->type(mod, spec_->type(mod), {});
}

/* Storage specifier */
inline ir::attribute_kind_t get_ir_attr(STORAGE_SPEC_T spec){
  switch(spec){
    case RESTRICT_T: return ir::noalias;
    case READONLY_T: return ir::readonly;
    case WRITEONLY_T: return ir::writeonly;
    default: throw std::runtime_error("cannot convert storage specifier to IR function attribute");
  }
}

void storage_specifier::add_attr(ir::function* fn, size_t pos) {
  fn->add_attr(pos, ir::attribute(get_ir_attr(value_)));
}

void storage_specifier::add_metadata(ir::module*, std::string) {
  throw std::runtime_error("storage specifier is not a metadata");
}

/* Alignment specifier */
void alignment_specifier::add_attr(ir::function* fn, size_t pos) {
  fn->add_attr(pos, ir::attribute(ir::aligned, cst_->value()));
}

void alignment_specifier::add_metadata(ir::module *mod, std::string name) {
  throw std::runtime_error("alignment specifier is not a metadata");
}

/* Multiple-Of specifier */
void multiple_of_specifier::add_attr(ir::function* fn, size_t pos) {
  fn->add_attr(pos, ir::attribute(ir::multiple_of, cst_->value()));
}

void multiple_of_specifier::add_metadata(ir::module *mod, std::string name) {
  mod->add_metadata(name, {ir::metadata::multiple_of, cst_->value()});
}


/* Function definition */
ir::value* function_definition::codegen(ir::module *mod) const{
  ir::function_type *prototype = (ir::function_type*)header_->type(mod, spec_->type(mod), spec_->modifiers());
  const std::string &name = header_->id()->name();
  ir::function *fn = mod->get_or_insert_function(name, prototype);
  for(unsigned i = 0; i < header_->get_num_args(); i++){
    parameter *param = header_->get_arg(i);
    std::vector<modifier*> modifiers = param->modifiers();
    for(modifier* m: modifiers)
      m->add_attr(fn, 1 + i);
  }
  header_->bind_parameters(mod, fn);
  ir::basic_block *entry = ir::basic_block::create(mod->get_context(), "entry", fn);
  mod->seal_block(entry);
  mod->get_builder().set_insert_point(entry);
  body_->codegen(mod);
  mod->get_builder().create_ret_void();
  return nullptr;
}

}

}
