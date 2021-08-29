#include "triton/codegen/pass.h"
#include "triton/driver/kernel.h"
#include "triton/driver/module.h"
#include "triton/driver/stream.h"
#include "triton/ir/builder.h"
#include "triton/ir/dispatch.h"
#include "triton/ir/enums.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/print.h"
#include <optional>
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex>
#include <string>
#include <sstream>

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
      }))
      .def("max_shared_memory", [](drv::cu_device *self) {
        return self->max_shared_memory();
      })
      .def("enable_peer_access", [](drv::cu_device *self, unsigned long long int peer_mem_ptr) {
        self->enable_peer_access(peer_mem_ptr);
      });
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

  py::class_<drv::cu_module, drv::module>(m, "cu_module")
      .def("ptx", &drv::cu_module::ptx)
      .def("cubin", [](drv::cu_module *self) { return py::bytes(self->cubin()); })
      .def("llir", &drv::cu_module::llir);

  py::class_<drv::kernel>(m, "kernel");
}

/*****************************************************************************/
/* Python bindings for triton::codegen                                       */
/*****************************************************************************/

void init_triton_codegen(py::module &&m) {
  m.def(
      "add_passes_to_emit_bin", [](ir::module &ir, drv::device *dev, int num_warps, int num_stages, bool force_nc_cache) {
        drv::module *mod;
        drv::kernel *ker;
        size_t shared_mem;
        triton::codegen::add_passes_to_emit_bin(ir, dev, num_warps, num_stages, force_nc_cache, mod, ker, shared_mem);
        std::stringstream ss;
        ir::print(ir, ss);
        return std::make_tuple(mod, ker, shared_mem, ss.str());
      },
      py::return_value_policy::take_ownership);
}

/*****************************************************************************/
/* User-facing language features                                             */
/*****************************************************************************/

void init_triton_frontend(py::module &&m) {
  using ret = py::return_value_policy;

  // programming model
  m.def("program_id", &ir::dispatch::program_id, ret::reference);
  m.def("num_programs", &ir::dispatch::num_programs, ret::reference);
  // binary
  m.def("add", &ir::dispatch::add, ret::reference);
  m.def("sub", &ir::dispatch::sub, ret::reference);
  m.def("mul", &ir::dispatch::mul, ret::reference);
  m.def("truediv", &ir::dispatch::truediv, ret::reference);
  m.def("floordiv", &ir::dispatch::floordiv, ret::reference);
  m.def("mod", &ir::dispatch::mod, ret::reference);
  m.def("and_", &ir::dispatch::and_, ret::reference);
  m.def("or_", &ir::dispatch::or_, ret::reference);
  m.def("xor_", &ir::dispatch::xor_, ret::reference);
  m.def("lshr", &ir::dispatch::lshr, ret::reference);
  m.def("shl", &ir::dispatch::shl, ret::reference);
  // unary
  m.def("plus", &ir::dispatch::plus, ret::reference);
  m.def("minus", &ir::dispatch::minus, ret::reference);
  m.def("invert", &ir::dispatch::invert, ret::reference);
  // comparison
  m.def("greater_than", &ir::dispatch::greater_than, ret::reference);
  m.def("greater_equal", &ir::dispatch::greater_equal, ret::reference);
  m.def("less_than", &ir::dispatch::less_than, ret::reference);
  m.def("less_equal", &ir::dispatch::less_equal, ret::reference);
  m.def("equal", &ir::dispatch::equal, ret::reference);
  m.def("not_equal", &ir::dispatch::not_equal, ret::reference);
  // block creation
  m.def("arange", &ir::dispatch::arange, ret::reference);
  m.def("zeros", &ir::dispatch::zeros, ret::reference);
  // type manipuatation
  m.def("reshape", &ir::dispatch::reshape, ret::reference);
  typedef std::tuple<ir::value *, ir::value *> (*broadcast_ty)(ir::value *, ir::value *, ir::builder *);
  typedef ir::value *(*broadcast_to_ty)(ir::value *, ir::type::block_shapes_t, ir::builder *);
  m.def("broadcast", (broadcast_ty)(&ir::dispatch::broadcast), ret::reference);
  m.def("broadcast_to", (broadcast_to_ty)(&ir::dispatch::broadcast), ret::reference);
  m.def("bitcast", &ir::dispatch::bitcast, ret::reference);
  m.def("cast", &ir::dispatch::cast, ret::reference);
  // memory
  m.def("load", &ir::dispatch::load, ret::reference);
  m.def("store", &ir::dispatch::store, ret::reference);
  m.def("atomic_cas", &ir::dispatch::atomic_cas, ret::reference);
  m.def("atomic_xchg", &ir::dispatch::atomic_xchg, ret::reference);
  m.def("atomic_add", &ir::dispatch::atomic_add, ret::reference);
  m.def("atomic_max", &ir::dispatch::atomic_max, ret::reference);
  m.def("atomic_min", &ir::dispatch::atomic_min, ret::reference);
  m.def("atomic_and", &ir::dispatch::atomic_and, ret::reference);
  m.def("atomic_or", &ir::dispatch::atomic_or, ret::reference);
  m.def("atomic_xor", &ir::dispatch::atomic_xor, ret::reference);
  // linear algebra
  m.def("dot", &ir::dispatch::dot, ret::reference);
  // indexing
  m.def("where", &ir::dispatch::where, ret::reference);
  // reduction
  m.def("min", &ir::dispatch::min, ret::reference);
  m.def("max", &ir::dispatch::max, ret::reference);
  m.def("sum", &ir::dispatch::sum, ret::reference);
  // math
  m.def("exp", &ir::dispatch::exp, ret::reference);
  m.def("log", &ir::dispatch::log, ret::reference);
  m.def("cos", &ir::dispatch::cos, ret::reference);
  m.def("sin", &ir::dispatch::sin, ret::reference);
  m.def("sqrt", &ir::dispatch::sqrt, ret::reference);
  // internal (debugging only)
  m.def("multiple_of", &ir::dispatch::multiple_of, ret::reference);
  m.def("max_contiguous", &ir::dispatch::max_contiguous, ret::reference);
  m.def("debug_barrier", &ir::dispatch::debug_barrier, ret::reference);
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

  py::class_<ir::instruction, ir::user>(m, "instruction");
  py::class_<ir::phi_node, ir::user>(m, "phi_node");

  py::class_<ir::type>(m, "type")
      .def("is_ptr", &ir::type::is_pointer_ty)
      .def("is_int", static_cast<bool (ir::type::*)() const>(&ir::type::is_integer_ty))
      .def("is_floating", &ir::type::is_floating_point_ty)
      .def("is_block", &ir::type::is_block_ty)
      .def("make_ptr", &ir::pointer_type::get, ret::reference)
      .def("make_function", &ir::function_type::get, ret::reference)
      .def("make_block", &ir::block_type::get, ret::reference)
      .def("get_void", &ir::type::get_void_ty, ret::reference)
      .def("get_fp8", &ir::type::get_fp8_ty, ret::reference)
      .def("get_fp16", &ir::type::get_fp16_ty, ret::reference)
      .def("get_bf16", &ir::type::get_bf16_ty, ret::reference)
      .def("get_fp32", &ir::type::get_fp32_ty, ret::reference)
      .def("get_fp64", &ir::type::get_fp64_ty, ret::reference)
      .def("get_int1", &ir::type::get_int1_ty, ret::reference)
      .def("get_int8", &ir::type::get_int8_ty, ret::reference)
      .def("get_int16", &ir::type::get_int16_ty, ret::reference)
      .def("get_int32", &ir::type::get_int32_ty, ret::reference)
      .def("get_int64", &ir::type::get_int64_ty, ret::reference)

      .def("is_void", &ir::type::is_void_ty)
      .def("is_fp8", &ir::type::is_fp8_ty)
      .def("is_fp16", &ir::type::is_fp16_ty)
      .def("is_bf16", &ir::type::is_bf16_ty)
      .def("is_fp32", &ir::type::is_fp32_ty)
      .def("is_fp64", &ir::type::is_fp64_ty)
      .def("is_int1", [](ir::type *self) { return self->is_integer_ty(1); })
      .def("is_int8", [](ir::type *self) { return self->is_integer_ty(8); })
      .def("is_int16", [](ir::type *self) { return self->is_integer_ty(16); })
      .def("is_int32", [](ir::type *self) { return self->is_integer_ty(32); })
      .def("is_int64", [](ir::type *self) { return self->is_integer_ty(64); })

      .def_property_readonly("fp_mantissa_width", &ir::type::get_fp_mantissa_width)
      .def_property_readonly("scalar", &ir::type::get_scalar_ty)
      .def_property_readonly("context", &ir::type::get_context, ret::reference);

  py::class_<ir::pointer_type, ir::type>(m, "pointer_type")
      .def_property_readonly("element", &ir::pointer_type::get_element_ty, ret::reference);

  py::class_<ir::function_type, ir::type>(m, "function_type");
  py::class_<ir::integer_type, ir::type>(m, "integer_type");
  py::class_<ir::block_type, ir::type>(m, "block_type")
      .def_property_readonly("shape", &ir::block_type::get_shapes)
      .def_property_readonly("numel", &ir::type::get_tile_num_elements);

  py::class_<ir::module>(m, "module")
      .def(py::init<std::string, ir::builder &>())
      .def("get_or_insert_function", &ir::module::get_or_insert_function, ret::reference)
      .def("seal_block", &ir::module::seal_block)
      .def("set_value", (void (ir::module::*)(const std::string &, ir::value *)) & ir::module::set_value)
      .def("set_type", &ir::module::set_type)
      .def("get_value", (ir::value * (ir::module::*)(const std::string &)) & ir::module::get_value, ret::reference)
      .def("get_values", &ir::module::get_values, ret::reference)
      .def("set_values", &ir::module::set_values)
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
      .def("get_int1", &ir::builder::get_int1, ret::reference)
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
