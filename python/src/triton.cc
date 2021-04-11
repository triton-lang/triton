#include "functions.h"
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
/* User-facing language features                                             */
/*****************************************************************************/

void init_triton_frontend(py::module &&m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;

  // triton.program_id
  DEF_FUNC(m, "program_id", program_id, "axis"_a);
  // triton.try_broadcast
  DEF_FUNC(m, "broadcast", try_broadcast, "input"_a, "other"_a);
  // triton.broadcast_to
  DEF_FUNC(m, "broadcast_to", broadcast_to, "input"_a, "shape"_a);
  // triton.dot
  DEF_FUNC(m, "dot", dot, "input"_a, "other"_a);
  // triton.arange
  DEF_FUNC(m, "arange", arange, "start"_a, "end"_a);
  // triton.zeros
  DEF_FUNC(m, "zeros", zeros, "shape"_a, "dtype"_a);
  // triton.load
  DEF_FUNC(m, "load", load, "pointer"_a, "mask"_a = py::none(), "other"_a = py::none());
  // triton.store
  DEF_FUNC(m, "store", store, "pointer"_a, "value"_a, "mask"_a = py::none());
  // triton.where
  DEF_FUNC(m, "where", where, "condition"_a, "x"_a, "y"_a);
  // triton.minimum
  DEF_FUNC(m, "minimum", minimum, "x"_a, "y"_a);

  auto py_array = py::class_<array>(m, "array");
  DEF_FUNC(py_array, "__add__", add, "other"_a);
  DEF_FUNC(py_array, "__sub__", sub, "other"_a);
  DEF_FUNC(py_array, "__mul__", mul, "other"_a);
  DEF_FUNC(py_array, "__div__", _div, "other"_a);
  DEF_FUNC(py_array, "__mod__", mod, "other"_a);
  DEF_FUNC(py_array, "__gt__", greater_than, "other"_a);
  DEF_FUNC(py_array, "__ge__", greater_equal, "other"_a);
  DEF_FUNC(py_array, "__lt__", less_than, "other"_a);
  DEF_FUNC(py_array, "__le__", less_equal, "other"_a);
  DEF_FUNC(py_array, "__and__", _and, "other"_a);
  DEF_FUNC(py_array, "__getitem__", subscript, "other"_a);
  DEF_FUNC(py_array, "to", cast, "dtype"_a);

  py::implicitly_convertible<ir::argument, array>();
  py::implicitly_convertible<ir::user, array>();
  //py::implicitly_convertible<array, ir::value>();
}

/*****************************************************************************/
/* Python bindings for triton::ir                                            */
/*****************************************************************************/

void init_triton_ir(py::module &&m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;

  py::class_<ir::context>(m, "context")
      .def(py::init<>());

  auto value = py::class_<ir::value>(m, "value");
  value.def_property("name", &ir::value::get_name, &ir::value::set_name);
  value.def_property_readonly("type", &ir::value::get_type);

  py::class_<ir::user, ir::value>(m, "user");

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

  py::class_<ir::builder>(m, "builder", py::dynamic_attr())
      .def(py::init<ir::context &>())
      // getters
      .def_property_readonly("context", &ir::builder::get_context, ret::reference)
      // control flow
      .def("br", &ir::builder::create_br, ret::reference)
      .def("cond_br", &ir::builder::create_cond_br, ret::reference)
      .def("ret_void", &ir::builder::create_ret_void, ret::reference)
      .def("get_insert_block", &ir::builder::get_insert_block, ret::reference)
      .def("set_insert_block", (void (ir::builder::*)(ir::basic_block *)) & ir::builder::set_insert_point)
      // constants
      .def("get_int32", &ir::builder::get_int32, ret::reference)
      .def("get_float16", &ir::builder::get_float16, ret::reference)
      .def("get_float32", &ir::builder::get_float32, ret::reference)
      .def("get_range", &ir::builder::get_range, ret::reference);
}

void init_triton(py::module &m) {
  py::module subm = m.def_submodule("triton");
  init_triton_codegen(std::move(subm.def_submodule("code_gen")));
  init_triton_driver(std::move(subm.def_submodule("driver")));
  init_triton_ir(std::move(subm.def_submodule("ir")));
  init_triton_frontend(std::move(subm.def_submodule("frontend")));
}
