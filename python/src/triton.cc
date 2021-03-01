#include "triton/driver/stream.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/lang/code_gen.h"
#include "triton/lang/cpp.h"
#include "triton/lang/parser.h"
#include "triton/runtime/arg.h"
#include "triton/runtime/function.h"
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex>
#include <string>

using namespace triton;
namespace rt = triton::runtime;
namespace drv = triton::driver;
namespace lng = triton::lang;

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
    if (std::find(names.begin(), names.end(), name) != names.end()) {
      std::string def = str.substr(pos, str.size() - pos);
      int count, pos;
      // skip over declaration
      // by finding matching ')' for first '('
      count = 1;
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
  }
  return ret;
}

/*****************************************************************************/
/* Python bindings for triton::tools                                         */
/*****************************************************************************/
void init_triton_tools(pybind11::module &&m) {
  m.def("extract_kernels", &extract_kernels);
}

/*****************************************************************************/
/* Python bindings for triton::driver                                        */
/*****************************************************************************/

void init_triton_driver(pybind11::module &&m) {
  // base device
  pybind11::class_<drv::device>(m, "device");
  // cuda device
  pybind11::class_<drv::cu_device, driver::device>(m, "cu_device")
      .def(pybind11::init<CUdevice, bool>());
  // host device
  pybind11::class_<drv::host_device, driver::device>(m, "host_device")
      .def(pybind11::init<>());

  // base stream
  pybind11::class_<drv::stream>(m, "stream");
  // cuda stream
  pybind11::class_<drv::host_stream, drv::stream>(m, "host_stream")
      .def(pybind11::init<>());
  // host stream
  pybind11::class_<drv::cu_stream, drv::stream>(m, "cu_stream")
      .def(pybind11::init([](uint64_t handle, bool take_ownership) {
        return std::unique_ptr<driver::cu_stream>(new driver::cu_stream((CUstream)handle, take_ownership));
      }));
}

/*****************************************************************************/
/* Python bindings for triton::runtime                                       */
/*****************************************************************************/
void init_triton_runtime(pybind11::module &&m) {
  // argument type
  pybind11::enum_<rt::arg_type>(m, "arg_type")
      .value("int1", rt::INT1_T)
      .value("int8", rt::INT8_T)
      .value("int16", rt::INT16_T)
      .value("int32", rt::INT32_T)
      .value("int64", rt::INT64_T)
      .value("half", rt::HALF_T)
      .value("float", rt::FLOAT_T)
      .value("double", rt::DOUBLE_T)
      .value("buffer", rt::BUFFER_T);
  // assembly mode
  pybind11::enum_<rt::asm_mode_t>(m, "asm_mode")
      .value("ptx", rt::ASM_NV_PTX)
      .value("sass", rt::ASM_NV_SASS);
  // compilation options
  pybind11::class_<rt::options_t>(m, "options", pybind11::dynamic_attr())
      .def(pybind11::init<>())
      .def_readwrite("defines", &rt::options_t::defines)
      .def_readwrite("num_warps", &rt::options_t::num_warps)
      .def("__getattr__", [](rt::options_t *opt, const std::string &name) {
        return opt->D<int>(name);
      });
  //  kernel
  pybind11::class_<rt::kernel>(m, "kernel")
      .def("__call__", &rt::kernel::operator())
      .def_readonly("opt", &rt::kernel::opt);
  // function
  pybind11::class_<rt::function>(m, "function")
      .def(pybind11::init<std::string, rt::options_t, driver::device *, rt::function::autotune_vals_t, std::vector<std::string>>())
      .def("autotune", &rt::function::autotune, pybind11::return_value_policy::reference_internal)
      .def("signature", &rt::function::get_signature);
}

void init_triton(pybind11::module &m) {
  pybind11::module subm = m.def_submodule("triton");
  init_triton_driver(std::move(subm.def_submodule("driver")));
  init_triton_runtime(std::move(subm.def_submodule("runtime")));
  init_triton_tools(std::move(subm.def_submodule("tools")));
}
