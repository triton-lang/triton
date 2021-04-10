#include "triton/codegen/pass.h"
#include "triton/driver/kernel.h"
#include "triton/driver/module.h"
#include "triton/driver/stream.h"
#include "triton/ir/builder.h"
#include "triton/ir/enums.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include <optional>
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex>
#include <string>

namespace py = pybind11;

namespace ir = triton::ir;
namespace drv = triton::driver;

/*****************************************************************************/
/* Python bindings for triton::driver                                        */
/*****************************************************************************/

void init_triton_driver(py::module &&m) {
  // base device
  py::class_<drv::device>(m, "device");
  // cuda device
  py::class_<drv::cu_device, drv::device>(m, "cu_device")
      .def(py::init([](int dev_id, bool take_ownership) {
        CUdevice handle;
        drv::dispatch::cuDeviceGet(&handle, dev_id);
        return new drv::cu_device(handle, take_ownership);
      }));
  // host device
  py::class_<drv::host_device, drv::device>(m, "host_device")
      .def(py::init<>());

  // base stream
  py::class_<drv::stream>(m, "stream");
  // host stream
  py::class_<drv::host_stream, drv::stream>(m, "host_stream")
      .def(py::init<>());
  // cuda stream
  py::class_<drv::cu_stream, drv::stream>(m, "cu_stream")
      // py doesn't support opaque pointer (e.g., CUstream) so
      // we assume it has been converted to uint64_t
      .def(py::init([](uint64_t handle, bool take_ownership) {
        return std::unique_ptr<drv::cu_stream>(new drv::cu_stream((CUstream)handle, take_ownership));
      }))
      .def("enqueue", [](drv::cu_stream *self, drv::kernel *kernel,
                         size_t grid_0, size_t grid_1, size_t grid_2,
                         size_t block_0, size_t block_1, size_t block_2,
                         const std::string &args,
                         size_t shared_mem) {
        return self->enqueue(kernel, {grid_0, grid_1, grid_2}, {block_0, block_1, block_2},
                             (void *)args.data(), args.size(), shared_mem);
      });

  py::class_<drv::module>(m, "module");
  //py::class_<drv::cu_module, drv::module>(m, "cu_module");

  py::class_<drv::kernel>(m, "kernel");
}

/*****************************************************************************/
/* Python bindings for triton::codegen                                       */
/*****************************************************************************/

void init_triton_codegen(py::module &&m) {
  m.def(
      "add_passes_to_emit_bin", [](ir::module &ir, drv::device *dev, int num_warps) {
        drv::module *mod;
        drv::kernel *ker;
        size_t shared_mem;
        triton::codegen::add_passes_to_emit_bin(ir, dev, num_warps, mod, ker, shared_mem);
        return std::make_tuple(mod, ker, shared_mem);
      },
      py::return_value_policy::take_ownership);
}

/*****************************************************************************/
/* Python bindings for triton::ir                                            */
/*****************************************************************************/

ir::value *add(ir::value *lhs, ir::value *rhs, ir::builder *builder) {
  ir::type *scalar_ty = lhs->get_type()->get_scalar_ty();
  // ptr + offset
  if (scalar_ty->is_pointer_ty())
    return builder->create_gep(lhs, {rhs});
  // float + float
  else if (scalar_ty->is_floating_point_ty())
    return builder->create_fadd(lhs, rhs);
  // int + int
  else if (scalar_ty->is_integer_ty())
    return builder->create_add(lhs, rhs);
}

ir::value *sub(ir::value *lhs, ir::value *rhs, ir::builder *builder) {
  ir::type *scalar_ty = lhs->get_type()->get_scalar_ty();
  // ptr + offset
  if (scalar_ty->is_pointer_ty())
    return builder->create_gep(lhs, {rhs});
  // float + float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fsub(lhs, rhs);
  // int + int
  else if (scalar_ty->is_integer_ty())
    return builder->create_sub(lhs, rhs);
}

ir::value *mul(ir::value *lhs, ir::value *rhs, ir::builder *builder) {
  ir::type *scalar_ty = lhs->get_type()->get_scalar_ty();
  // float * float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fmul(lhs, rhs);
  // int * int
  else if (scalar_ty->is_integer_ty())
    return builder->create_mul(lhs, rhs);
}

ir::value *greater_than(ir::value *lhs, ir::value *rhs, ir::builder *builder) {
  ir::type *scalar_ty = lhs->get_type()->get_scalar_ty();
  // float > float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fcmpOGT(lhs, rhs);
  // int > int
  else if (scalar_ty->is_integer_ty())
    return builder->create_icmpSGT(lhs, rhs);
}

ir::value *greater_equal(ir::value *lhs, ir::value *rhs, ir::builder *builder) {
  ir::type *scalar_ty = lhs->get_type()->get_scalar_ty();
  // float >= float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fcmpOGE(lhs, rhs);
  // int >= int
  else if (scalar_ty->is_integer_ty())
    return builder->create_icmpSGE(lhs, rhs);
}

ir::value *less_than(ir::value *lhs, ir::value *rhs, ir::builder *builder) {
  ir::type *scalar_ty = lhs->get_type()->get_scalar_ty();
  // float < float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fcmpOLT(lhs, rhs);
  // int < int
  else if (scalar_ty->is_integer_ty())
    return builder->create_icmpSLT(lhs, rhs);
}

ir::value *div_(ir::value *lhs, ir::value *rhs, ir::builder *builder) {
  ir::type *scalar_ty = lhs->get_type()->get_scalar_ty();
  // float / float
  if (scalar_ty->is_floating_point_ty())
    return builder->create_fdiv(lhs, rhs);
  // int / int
  else if (scalar_ty->is_integer_ty())
    return builder->create_sdiv(lhs, rhs);
}

ir::value *mod(ir::value *lhs, ir::value *rhs, ir::builder *builder) {
  ir::type *scalar_ty = lhs->get_type()->get_scalar_ty();
  // float % int
  if (scalar_ty->is_floating_point_ty())
    return builder->create_frem(lhs, rhs);
  // int % int
  else if (scalar_ty->is_integer_ty())
    return builder->create_srem(lhs, rhs);
}

ir::value *and_(ir::value *lhs, ir::value *rhs, ir::builder *builder) {
  return builder->create_and(lhs, rhs);
}

ir::value *convert(ir::value *arg, ir::type *type, ir::builder *builder) {
  if (type->is_block_ty())
    type = ir::block_type::get(type, arg->get_type()->get_block_shapes());
  // FP Truncation
  ir::type *src_ty = arg->get_type()->get_scalar_ty();
  ir::type *dst_ty = type->get_scalar_ty();
  bool truncate_fp = src_ty->is_floating_point_ty() &&
                     dst_ty->is_floating_point_ty() &&
                     src_ty->get_fp_mantissa_width() > dst_ty->get_fp_mantissa_width();
  if (truncate_fp)
    return builder->create_fp_trunc(arg, type);
}

ir::value *broadcast_to(ir::value *arg0, const ir::type::block_shapes_t &shape, ir::builder *builder) {
  if (!arg0->get_type()->is_block_ty())
    return builder->create_splat(arg0, shape);
  auto src_shape = arg0->get_type()->get_block_shapes();
  if (src_shape.size() != shape.size())
    throw std::runtime_error("Cannot broadcast");
  return builder->create_broadcast(arg0, shape);
}

std::tuple<ir::value *, ir::value *> broadcast(ir::value *lhs, ir::value *rhs, ir::builder *builder) {
  ir::type *lhs_ty = lhs->get_type();
  ir::type *rhs_ty = rhs->get_type();
  // broadcast(block, scalar)
  if (lhs_ty->is_block_ty() && !rhs_ty->is_block_ty())
    rhs = builder->create_splat(rhs, lhs_ty->get_block_shapes());
  // broadcast(scalar, block)
  else if (!lhs_ty->is_block_ty() && rhs_ty->is_block_ty())
    lhs = builder->create_splat(lhs, rhs_ty->get_block_shapes());
  // broadcast(block, block)
  else if (lhs_ty->is_block_ty() && rhs_ty->is_block_ty()) {
    auto lhs_shape = lhs_ty->get_block_shapes();
    auto rhs_shape = rhs_ty->get_block_shapes();
    if (lhs_shape.size() != rhs_shape.size())
      throw std::runtime_error("Cannot broadcast: blocks must have the same rank");
    ir::type::block_shapes_t ret_shape;
    for (size_t i = 0; i < lhs_shape.size(); ++i) {
      unsigned left = lhs_shape[i];
      unsigned right = rhs_shape[i];
      if (left == 1)
        ret_shape.push_back(right);
      else if (right == 1)
        ret_shape.push_back(left);
      else if (left == right)
        ret_shape.push_back(left);
      else
        throw std::runtime_error("Cannot broadcast: incompatible dimensions at index " + std::to_string(i) +
                                 ": " + std::to_string(left) + " and " + std::to_string(right));
    }
    if (lhs_shape != ret_shape)
      lhs = broadcast_to(lhs, ret_shape, builder);
    if (rhs_shape != ret_shape)
      rhs = broadcast_to(rhs, ret_shape, builder);
  }
  return std::make_tuple(lhs, rhs);
}

ir::value *subscript(ir::value *lhs, std::vector<py::object> slices, ir::builder *builder) {
  ir::type::block_shapes_t shape;
  size_t curr = 0;
  for (py::object slice : slices) {
    py::object none = py::none();
    py::object all = py::make_tuple(none, none, none);
    if (slice.is(none))
      shape.push_back(1);
    else if (slice.is(all))
      shape.push_back(lhs->get_type()->get_block_shapes()[curr]);
    else
      throw std::runtime_error("unsupported slice");
  }
  return builder->create_reshape(lhs, shape);
}

ir::value *min(ir::value *lhs, ir::value *rhs, ir::builder *builder) {
  ir::value *cond = less_than(lhs, rhs, builder);
  return builder->create_select(cond, lhs, rhs);
}

ir::value *load(ir::value *ptr, std::optional<ir::value *> _mask, std::optional<ir::value *> _else_value, ir::builder *builder) {
  if (!_mask.has_value() && !_else_value.has_value())
    return builder->create_load(ptr);
  if (!_mask.has_value())
    throw std::runtime_error("`else_value` cannot be provided without `mask`");
  ir::value *mask = _mask.value();
  ir::type *elt_ty = ptr->get_type()->get_scalar_ty()->get_pointer_element_ty();
  ir::value *else_value = _else_value.has_value() ? _else_value.value() : ir::undef_value::get(elt_ty);
  else_value = convert(else_value, elt_ty, builder);
  else_value = broadcast_to(else_value, ptr->get_type()->get_block_shapes(), builder);
  return builder->create_masked_load(ptr, mask, else_value);
}

ir::value *store(ir::value *ptr, ir::value *val, std::optional<ir::value *> _mask, ir::builder *builder) {
  if (!_mask.has_value())
    return builder->create_store(ptr, val);
  ir::value *mask = _mask.value();
  return builder->create_masked_store(ptr, val, mask);
}

ir::value *dot(ir::value *lhs, ir::value *rhs, ir::builder *builder) {
  ir::value *_0 = builder->get_float32(0);
  _0 = builder->create_splat(_0, lhs->get_type()->get_block_shapes());
  return builder->create_dot(lhs, rhs, _0);
}

ir::value *select_(ir::value *cond, ir::value *true_val, ir::value *false_val, ir::builder *builder) {
  return builder->create_select(cond, true_val, false_val);
}

ir::value *arange(int start, int end, ir::builder *builder) {
  return builder->get_range(start, end);
}

ir::value *program_id(int axis, ir::builder *builder) {
  return builder->create_get_program_id(axis);
}

ir::value *zeros(ir::type::block_shapes_t shape, ir::type *dtype, ir::builder *builder) {
  ir::value *_0 = ir::constant::get_null_value(dtype);
  return builder->create_splat(_0, shape);
}

void init_triton_ir(py::module &&m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;

  py::class_<ir::context>(m, "context")
      .def(py::init<>());

  py::class_<ir::value>(m, "value")
      .def_property("name", &ir::value::get_name, &ir::value::set_name)
      .def_property_readonly("type", &ir::value::get_type);

  py::class_<ir::user, ir::value>(m, "user")
      .def("__add__", &add, ret::reference)
      .def("__sub__", &sub, ret::reference)
      .def("__mul__", &mul, ret::reference)
      .def("__div__", &div_, ret::reference)
      .def("__mod__", &mod, ret::reference)
      .def("__gt__", &greater_than, ret::reference)
      .def("__ge__", &greater_equal, ret::reference)
      .def("__lt__", &less_than, ret::reference)
      .def("__and__", &and_, ret::reference)
      .def("__getitem__", &subscript, ret::reference)
      .def("to", &convert, ret::reference);

  m.def("broadcast", &broadcast, ret::reference);
  m.def("broadcast_to", &broadcast_to, ret::reference);
  m.def("min", &min, ret::reference);
  m.def("load", &load, ret::reference);
  m.def("store", &store, ret::reference);
  m.def("dot", &dot, ret::reference);
  m.def("select", &select_, ret::reference);
  m.def("arange", &arange, ret::reference);
  m.def("program_id", &program_id, ret::reference);
  m.def("zeros", &zeros, ret::reference);

  py::class_<ir::constant, ir::user>(m, "constant");

  py::class_<ir::undef_value, ir::constant>(m, "undef")
      .def("get", &ir::undef_value::get, ret::reference);

  py::class_<ir::constant_int, ir::constant>(m, "constant_int")
      .def_property_readonly("value", &ir::constant_int::get_value)
      .def("__int__", [](ir::constant_int *self) { return self->get_value(); });

  py::class_<ir::constant_fp, ir::constant>(m, "constant_float")
      .def_property_readonly("value", &ir::constant_fp::get_value);

  py::class_<ir::type>(m, "type")
      .def("is_ptr", &ir::type::is_pointer_ty)
      .def("is_int", static_cast<bool (ir::type::*)() const>(&ir::type::is_integer_ty))
      .def("is_floating", &ir::type::is_floating_point_ty)
      .def("is_block", &ir::type::is_block_ty)
      .def("make_ptr", &ir::pointer_type::get, ret::reference)
      .def("make_function", &ir::function_type::get, ret::reference)
      .def("make_block", &ir::block_type::get, ret::reference)
      .def("get_void", &ir::type::get_void_ty, ret::reference)
      .def("get_fp16", &ir::type::get_half_ty, ret::reference)
      .def("get_fp32", &ir::type::get_float_ty, ret::reference)
      .def("get_fp64", &ir::type::get_double_ty, ret::reference)
      .def("get_int1", &ir::type::get_int1_ty, ret::reference)
      .def("get_int8", &ir::type::get_int8_ty, ret::reference)
      .def("get_int16", &ir::type::get_int16_ty, ret::reference)
      .def("get_int32", &ir::type::get_int32_ty, ret::reference)
      .def("get_int64", &ir::type::get_int64_ty, ret::reference)
      .def_property_readonly("fp_mantissa_width", &ir::type::get_fp_mantissa_width)
      .def_property_readonly("scalar", &ir::type::get_scalar_ty)
      .def_property_readonly("context", &ir::type::get_context, ret::reference);

  py::class_<ir::pointer_type, ir::type>(m, "pointer_type")
      .def_property_readonly("element", &ir::pointer_type::get_element_ty, ret::reference);

  py::class_<ir::function_type, ir::type>(m, "function_type");
  py::class_<ir::integer_type, ir::type>(m, "integer_type");
  py::class_<ir::block_type, ir::type>(m, "block_type")
      .def_property_readonly("shape", &ir::block_type::get_shapes);

  py::class_<ir::scope>(m, "scope")
      .def(py::init<>())
      .def_property_readonly("values", &ir::scope::get_values)
      .def("set_type", &ir::scope::set_type);

  py::class_<ir::module>(m, "module")
      .def(py::init<std::string, ir::builder &>())
      .def("get_or_insert_function", &ir::module::get_or_insert_function, ret::reference)
      .def("add_new_scope", &ir::module::add_new_scope, ret::reference)
      .def("seal_block", &ir::module::seal_block)
      .def("set_value", (void (ir::module::*)(const std::string &, ir::value *)) & ir::module::set_value)
      .def("get_value", (ir::value * (ir::module::*)(const std::string &)) & ir::module::get_value, ret::reference)
      .def("pop_scope", &ir::module::pop_scope)
      .def_property_readonly("scope", &ir::module::get_scope, ret::reference)
      .def_property_readonly("builder", &ir::module::get_builder, ret::reference);

  using eattr = ir::attribute_kind_t;
  py::enum_<eattr>(m, "attribute_kind")
      .value("readonly", eattr::readonly)
      .value("writeonly", eattr::writeonly)
      .value("noalias", eattr::noalias)
      .value("aligned", eattr::aligned)
      .value("multiple_of", eattr::multiple_of)
      .value("retune", eattr::retune)
      .value("not_implemented", eattr::not_implemented);

  py::class_<ir::attribute>(m, "attribute")
      .def(py::init<eattr, int>());

  py::class_<ir::function>(m, "function")
      .def_property_readonly("args", &ir::function::args)
      .def_property_readonly("attrs", &ir::function::attrs)
      .def("add_attr", &ir::function::add_attr);

  py::class_<ir::argument, ir::value>(m, "argument");

  py::class_<ir::basic_block, ir::value>(m, "basic_block")
      .def("create", &ir::basic_block::create, ret::reference)
      .def_property_readonly("parent", &ir::basic_block::get_parent, ret::reference);

  using ecast = ir::cast_op_t;
  py::enum_<ir::cast_op_t>(m, "cast_op")
      .value("trunc", ecast::Trunc)
      .value("zext", ecast::ZExt)
      .value("sext", ecast::SExt)
      .value("fptrunc", ecast::FPTrunc)
      .value("fpext", ecast::FPExt)
      .value("uitofp", ecast::UIToFP)
      .value("sitofp", ecast::SIToFP)
      .value("fptoui", ecast::FPToUI)
      .value("fptosi", ecast::FPToSI)
      .value("ptrtoint", ecast::PtrToInt)
      .value("inttoptr", ecast::IntToPtr)
      .value("bitcast", ecast::BitCast)
      .value("addrspacecast", ecast::AddrSpaceCast);

  py::class_<ir::builder>(m, "builder", py::dynamic_attr())
      .def(py::init<ir::context &>())
      // getters
      .def_property_readonly("context", &ir::builder::get_context, ret::reference)
      // terminator instructions
      .def("br", &ir::builder::create_br, ret::reference)
      .def("cond_br", &ir::builder::create_cond_br, ret::reference)
      .def("ret_void", &ir::builder::create_ret_void, ret::reference)
      // Cast instructions
      .def("cast", &ir::builder::create_cast, ret::reference)
      .def("ptr_to_int", &ir::builder::create_ptr_to_int, ret::reference)
      .def("si_to_fp", &ir::builder::create_si_to_fp, ret::reference)
      .def("ui_to_fp", &ir::builder::create_ui_to_fp, ret::reference)
      .def("fp_to_si", &ir::builder::create_fp_to_si, ret::reference)
      .def("fp_to_ui", &ir::builder::create_fp_to_ui, ret::reference)
      .def("fp_ext", &ir::builder::create_fp_ext, ret::reference)
      .def("fp_trunc", &ir::builder::create_fp_trunc, ret::reference)
      .def("int_cast", &ir::builder::create_int_cast, ret::reference)
      .def("downcast", &ir::builder::create_downcast, ret::reference)
      // Binary instructions
      .def("insert_nuwnswb_binop", &ir::builder::create_insert_nuwnswb_binop, ret::reference)
      .def("fmul", &ir::builder::create_fmul, ret::reference)
      .def("fdiv", &ir::builder::create_fdiv, ret::reference)
      .def("frem", &ir::builder::create_frem, ret::reference)
      .def("fadd", &ir::builder::create_fadd, ret::reference)
      .def("fsub", &ir::builder::create_fsub, ret::reference)
      .def("mul", &ir::builder::create_mul, ret::reference, "lhs"_a, "rhs"_a, "has_nuw"_a = false, "has_nsw"_a = false)
      .def("sdiv", &ir::builder::create_sdiv, ret::reference)
      .def("udiv", &ir::builder::create_udiv, ret::reference)
      .def("srem", &ir::builder::create_srem, ret::reference)
      .def("urem", &ir::builder::create_urem, ret::reference)
      .def("add", &ir::builder::create_add, ret::reference, "lhs"_a, "rhs"_a, "has_nuw"_a = false, "has_nsw"_a = false)
      .def("sub", &ir::builder::create_sub, ret::reference, "lhs"_a, "rhs"_a, "has_nuw"_a = false, "has_nsw"_a = false)
      .def("shl", &ir::builder::create_shl, ret::reference, "lhs"_a, "rhs"_a, "has_nuw"_a = false, "has_nsw"_a = false)
      .def("lshr", &ir::builder::create_lshr, ret::reference, "lhs"_a, "rhs"_a, "has_nuw"_a = false, "has_nsw"_a = false)
      .def("ashr", &ir::builder::create_ashr, ret::reference, "lhs"_a, "rhs"_a, "has_nuw"_a = false, "has_nsw"_a = false)
      // GEP
      .def("gep", &ir::builder::create_gep, ret::reference)
      // Comparison (int, ret::reference)
      .def("icmp", &ir::builder::create_icmp, ret::reference)
      .def("icmpSLE", &ir::builder::create_icmpSLE, ret::reference)
      .def("icmpSLT", &ir::builder::create_icmpSLT, ret::reference)
      .def("icmpSGE", &ir::builder::create_icmpSGE, ret::reference)
      .def("icmpSGT", &ir::builder::create_icmpSGT, ret::reference)
      .def("icmpULE", &ir::builder::create_icmpULE, ret::reference)
      .def("icmpULT", &ir::builder::create_icmpULT, ret::reference)
      .def("icmpUGE", &ir::builder::create_icmpUGE, ret::reference)
      .def("icmpUGT", &ir::builder::create_icmpUGT, ret::reference)
      .def("icmpEQ", &ir::builder::create_icmpEQ, ret::reference)
      .def("icmpNE", &ir::builder::create_icmpNE, ret::reference)
      // Comparison (float, ret::reference)
      .def("fcmp", &ir::builder::create_fcmp, ret::reference)
      .def("fcmpOLT", &ir::builder::create_fcmpOLT, ret::reference)
      .def("fcmpOGT", &ir::builder::create_fcmpOGT, ret::reference)
      .def("fcmpOLE", &ir::builder::create_fcmpOLE, ret::reference)
      .def("fcmpOGE", &ir::builder::create_fcmpOGE, ret::reference)
      .def("fcmpOEQ", &ir::builder::create_fcmpOEQ, ret::reference)
      .def("fcmpONE", &ir::builder::create_fcmpONE, ret::reference)
      // Logical
      .def("and_", &ir::builder::create_and, ret::reference)
      .def("xor", &ir::builder::create_xor, ret::reference)
      .def("or", &ir::builder::create_or, ret::reference)
      // Unary
      //  .def("fneg", &ir::builder::create_fneg, ret::reference)
      //  .def("neg", &ir::builder::create_neg, ret::reference)
      //  .def("not", &ir::builder::create_not, ret::reference)
      // Input/Output
      .def("load", &ir::builder::create_load, ret::reference)
      .def("store", &ir::builder::create_store, ret::reference)
      .def("masked_load", &ir::builder::create_masked_load, ret::reference)
      .def("masked_store", &ir::builder::create_masked_store, ret::reference)
      // Tile instruction
      .def("splat", &ir::builder::create_splat, ret::reference)
      .def("reshape", &ir::builder::create_reshape, ret::reference)
      .def("broadcast", &ir::builder::create_broadcast, ret::reference)
      // Built-in instruction
      .def("get_program_id", &ir::builder::create_get_program_id, ret::reference)
      .def("get_num_program", &ir::builder::create_get_num_program, ret::reference)
      .def("atomic_cas", &ir::builder::create_atomic_cas, ret::reference)
      .def("atomic_exch", &ir::builder::create_atomic_exch, ret::reference)
      .def("atomic_add", &ir::builder::create_atomic_add, ret::reference)
      .def("exp", &ir::builder::create_exp, ret::reference)
      .def("log", &ir::builder::create_log, ret::reference)
      .def("dot", &ir::builder::create_dot, ret::reference)
      .def("trans", &ir::builder::create_trans, ret::reference)
      .def("sqrt", &ir::builder::create_sqrt, ret::reference)
      .def("reduce", &ir::builder::create_reduce, ret::reference)
      .def("select", &ir::builder::create_select, ret::reference)
      // constants
      .def("get_int32", &ir::builder::get_int32, ret::reference)
      .def("get_float16", &ir::builder::get_float16, ret::reference)
      .def("get_float32", &ir::builder::get_float32, ret::reference)
      .def("get_range", &ir::builder::get_range, ret::reference)
      // control-flow
      .def("get_insert_block", &ir::builder::get_insert_block, ret::reference)
      .def("set_insert_block", (void (ir::builder::*)(ir::basic_block *)) & ir::builder::set_insert_point);
}

void init_triton(py::module &m) {
  py::module subm = m.def_submodule("triton");
  init_triton_codegen(std::move(subm.def_submodule("codegen")));
  init_triton_driver(std::move(subm.def_submodule("driver")));
  init_triton_ir(std::move(subm.def_submodule("ir")));
}
