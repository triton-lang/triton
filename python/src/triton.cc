#include "triton/codegen/pass.h"
#include "triton/driver/kernel.h"
#include "triton/driver/module.h"
#include "triton/driver/stream.h"
#include "triton/ir/builder.h"
#include "triton/ir/enums.h"
#include "triton/ir/module.h"
#include "triton/runtime/function.h"
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex>
#include <string>

namespace py = pybind11;

using namespace triton;
namespace ir = triton::ir;
namespace rt = triton::runtime;
namespace drv = triton::driver;

/*****************************************************************************/
/* Python bindings for triton::tools                                         */
/*****************************************************************************/

/*!
  @brief Function for extracting kernels out of a given source-string

  This can be important to enable pre-processor macros (or tunable parameters) that should only
  be defined within the scope of a single kernel function
*/
std::string extract_kernels(const std::string &str, const std::vector<std::string> &names) {
  if (names.empty())
    return str;
  // search for all regex matches of kernel_regex in str
  std::smatch matches;
  std::regex regex(" *__global__ +void +([_a-zA-Z][_a-zA-Z0-9]{0,30})");
  std::sregex_iterator it(str.begin(), str.end(), regex);
  std::sregex_iterator end;
  std::vector<std::tuple<std::string, int, int>> kernels;
  for (; it != end; ++it) {
    int pos = it->position();
    int len = it->length();
    std::string name = it->str(1);
    kernels.push_back(std::make_tuple(name, pos, len));
  }
  // check that all the kernels provided actually exist
  for (const std::string &name : names) {
    auto pred = [&name](const std::tuple<std::string, int, int> &t) { return std::get<0>(t) == name; };
    bool found = std::any_of(kernels.begin(), kernels.end(), pred);
    if (!found)
      throw std::runtime_error("Unable to find kernel `" + name + "` in provided source code:\n" + str);
  }
  // simple parsing logic to extract the declaration and body of each specified kernel
  std::string ret;
  for (const auto &k : kernels) {
    std::string name;
    int pos, len;
    std::tie(name, pos, len) = k;
    if (std::find(names.begin(), names.end(), name) == names.end())
      continue;
    std::string def = str.substr(pos, str.size() - pos);
    // skip over declaration
    // by finding matching ')' for first '('
    int count = 1;
    pos = def.find('(');
    while (!(def[pos++] == ')' && count == 0) && pos < def.size()) {
      count += def[pos] == '(';
      count -= def[pos] == ')';
    }
    // skip over definition
    // by finding matching '{' for first '}'
    count = 1;
    pos = def.find('{', pos);
    while (!(def[pos++] == '}' && count == 0) && pos < def.size()) {
      count += def[pos] == '{';
      count -= def[pos] == '}';
    }
    ret += def.substr(0, pos);
    ret += '\n';
  }
  return ret;
}

void init_triton_tools(py::module &&m) {
  m.def("extract_kernels", &extract_kernels);
}

/*****************************************************************************/
/* Python bindings for triton::driver                                        */
/*****************************************************************************/

void init_triton_driver(py::module &&m) {
  // base device
  py::class_<drv::device>(m, "device");
  // cuda device
  py::class_<drv::cu_device, driver::device>(m, "cu_device")
      .def(py::init([](int dev_id, bool take_ownership) {
        CUdevice handle;
        drv::dispatch::cuDeviceGet(&handle, dev_id);
        return new drv::cu_device(handle, take_ownership);
      }));
  // host device
  py::class_<drv::host_device, driver::device>(m, "host_device")
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
        return std::unique_ptr<driver::cu_stream>(new driver::cu_stream((CUstream)handle, take_ownership));
      }))
      .def("enqueue", [](driver::cu_stream *self, driver::kernel *kernel,
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
/* Python bindings for triton::runtime                                       */
/*****************************************************************************/
void init_triton_runtime(py::module &&m) {
  // argument type
  py::enum_<rt::arg_type>(m, "arg_type")
      .value("int1", rt::INT1_T)
      .value("int8", rt::INT8_T)
      .value("int16", rt::INT16_T)
      .value("int32", rt::INT32_T)
      .value("int64", rt::INT64_T)
      .value("half", rt::HALF_T)
      .value("float", rt::FLOAT_T)
      .value("double", rt::DOUBLE_T)
      .value("buffer", rt::BUFFER_T);
  // compilation options
  py::class_<rt::options_t>(m, "options", py::dynamic_attr())
      .def(py::init<>())
      .def_readwrite("defines", &rt::options_t::defines)
      .def_readwrite("num_warps", &rt::options_t::num_warps)
      .def("__getattr__", [](rt::options_t *opt, const std::string &name) {
        return opt->D<int>(name);
      });
  //  kernel
  py::class_<rt::kernel>(m, "kernel")
      .def("__call__", &rt::kernel::operator())
      .def_readonly("opt", &rt::kernel::opt)
      .def("asm", &rt::kernel::get_asm);
  // tune conf
  py::class_<rt::config>(m, "config")
      .def(py::init<std::map<std::string, std::string>, int>(),
           py::arg("defines") = std::map<std::string, std::string>(),
           py::arg("num_warps"));

  // function
  py::class_<rt::function>(m, "function")
      .def(py::init<const std::string &, const rt::options_t &, driver::device *, const std::vector<rt::config> &, const std::vector<std::string> &>())
      .def("autotune", &rt::function::autotune, py::return_value_policy::reference_internal)
      .def("signature", &rt::function::get_signature);
}

/*****************************************************************************/
/* Python bindings for triton::codegen                                       */
/*****************************************************************************/
void init_triton_codegen(py::module &&m) {
  m.def(
      "add_passes_to_emit_bin", [](ir::module &ir, driver::device *dev, int num_warps) {
        driver::module *mod;
        driver::kernel *ker;
        size_t shared_mem;
        triton::codegen::add_passes_to_emit_bin(ir, dev, num_warps, mod, ker, shared_mem);
        return std::make_tuple(mod, ker, shared_mem);
      },
      py::return_value_policy::take_ownership);
}

/*****************************************************************************/
/* Python bindings for triton::ir                                            */
/*****************************************************************************/
ir::builder *get_builder(ir::value *lhs, ir::value *rhs) {
  return lhs->get_type()->get_context().builder;
};

// ir::value *__add__(ir::value *lhs, ir::value *rhs) {
//   ir::builder *builder = get_builder(lhs, rhs);
//   ir::type *lhs_ty = lhs->get_type()->get_scalar_ty();
//   ir::type *rhs_ty = rhs->get_type()->get_scalar_ty();
//   // ptr + offset
//   if (lhs_ty->is_pointer_ty())
//     return builder->create_gep(lhs, {rhs});
//   // float + float
//   else if (lhs_ty->is_floating_point_ty())
//     return builder->create_fadd(lhs, rhs);
//   // int + int
//   else if (lhs_ty->is_integer_ty())
//     return builder->create_add(lhs, rhs);
//   throw pybind11::value_error("unsupported operands for +");
// }

// ir::value *__mul__(ir::value *lhs, ir::value *rhs) {
//   ir::builder *builder = get_builder(lhs, rhs);
//   ir::type *lhs_ty = lhs->get_type()->get_scalar_ty();
//   ir::type *rhs_ty = rhs->get_type()->get_scalar_ty();
//   // float * float
//   if (lhs_ty->is_floating_point_ty())
//     return builder->create_fmul(lhs, rhs);
//   // int * int
//   else if (lhs_ty->is_integer_ty())
//     return builder->create_mul(lhs, rhs);
//   throw pybind11::value_error("unsupported operands for *");
// }

// ir::value *__gt__(ir::value *lhs, ir::value *rhs) {
//   ir::builder *builder = get_builder(lhs, rhs);
//   ir::type *lhs_ty = lhs->get_type()->get_scalar_ty();
//   ir::type *rhs_ty = rhs->get_type()->get_scalar_ty();
//   // float > float
//   if (lhs_ty->is_floating_point_ty())
//     return builder->create_fcmpOGT(lhs, rhs);
//   else if (lhs_ty->is_integer_ty())
//     return builder->create_icmpSGT(lhs, rhs);
//   throw pybind11::value_error("unsupported operands for >");
// }

// ir::value *__lt__(ir::value *lhs, ir::value *rhs) {
//   ir::builder *builder = get_builder(lhs, rhs);
//   ir::type *lhs_ty = lhs->get_type()->get_scalar_ty();
//   ir::type *rhs_ty = rhs->get_type()->get_scalar_ty();
//   // float > float
//   if (lhs_ty->is_floating_point_ty())
//     return builder->create_fcmpOLT(lhs, rhs);
//   else if (lhs_ty->is_integer_ty())
//     return builder->create_icmpSLT(lhs, rhs);
//   throw pybind11::value_error("unsupported operands for <");
// }

void init_triton_ir(py::module &&m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;

  py::class_<ir::context>(m, "context")
      .def(py::init<>())
      .def_readwrite("builder", &ir::context::builder);

  py::class_<ir::value>(m, "value")
      .def_property("name", &ir::value::get_name, &ir::value::set_name)
      .def_property_readonly("type", &ir::value::get_type);
  // .def("__add__", &__add__, ret::reference)
  // .def("__mul__", &__mul__, ret::reference)
  // .def("__gt__", &__gt__, ret::reference)
  // .def("__lt__", &__lt__, ret::reference);

  py::class_<ir::user, ir::value>(m, "user");

  py::class_<ir::constant, ir::user>(m, "constant");

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

  py::class_<ir::pointer_type, ir::type>(m, "pointer_type");
  py::class_<ir::function_type, ir::type>(m, "function_type");
  py::class_<ir::integer_type, ir::type>(m, "integer_type");
  py::class_<ir::block_type, ir::type>(m, "block_type")
      .def_property_readonly("shape", &ir::block_type::get_shapes);

  py::class_<ir::scope>(m, "scope")
      .def(py::init<>())
      .def_property_readonly("values", &ir::scope::get_values)
      .def("set_type", &ir::scope::set_type);

  py::class_<ir::module>(m, "module")
      .def(py::init<std::string>())
      .def("get_or_insert_function", &ir::module::get_or_insert_function, ret::reference)
      .def("add_new_scope", &ir::module::add_new_scope, ret::reference)
      .def("seal_block", &ir::module::seal_block)
      .def("set_value", (void (ir::module::*)(const std::string &, ir::value *)) & ir::module::set_value)
      .def("get_value", (ir::value * (ir::module::*)(const std::string &)) & ir::module::get_value, ret::reference)
      .def("pop_scope", &ir::module::pop_scope)
      .def_property_readonly("context", &ir::module::get_context, ret::reference)
      .def_property_readonly("scope", &ir::module::get_scope, ret::reference)
      .def_property_readonly("builder", &ir::module::get_builder);

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

  py::class_<ir::builder>(m, "builder")
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
      .def("and", &ir::builder::create_and, ret::reference)
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
  init_triton_runtime(std::move(subm.def_submodule("runtime")));
  init_triton_tools(std::move(subm.def_submodule("tools")));
}
