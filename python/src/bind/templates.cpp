/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include "isaac/templates/common.hpp"
#include "isaac/templates/error.hpp"
#include "isaac/templates/conv.h"
#include "isaac/templates/gemm.h"
#include "isaac/driver/buffer.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/stream.h"


namespace py = pybind11;
namespace sc = isaac;
namespace tpt = sc::templates;

using sc::param_t;

template<class FUN>
py::array_t<bool> check_valid(sc::driver::Device const & device, py::buffer b){
  py::buffer_info info = b.request();
  auto result = py::array_t<uint8_t>(info.shape[0]);
  FUN::check_valid(device, info.shape[0], (param_t*)info.ptr, (uint8_t*)result.request().ptr);
  return result;
}

void export_templates(py::module&& m){


  py::enum_<sc::IsaacOperation_t>(m, "op")
      .value("OP_N", sc::IsaacOperation_t::ISAAC_OP_N)
      .value("OP_T", sc::IsaacOperation_t::ISAAC_OP_T)
      .export_values();


  py::class_<tpt::GEMM>(m, "GEMM")
      .def(py::init<sc::DType,sc::IsaacOperation_t,sc::IsaacOperation_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,
                    param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t>())
      .def_static("check_valid", check_valid<tpt::GEMM>)
      .def("dump", &tpt::GEMM::dump)
      .def("enqueue", &tpt::GEMM::enqueue);

  py::class_<tpt::Conv>(m, "Conv")
      .def(py::init<sc::DType, param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,
                    param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t>())
      .def_static("check_valid", check_valid<tpt::Conv>)
      .def("dump", &tpt::Conv::dump)
      .def("enqueue", &tpt::Conv::enqueue);


  py::register_exception<tpt::invalid_parameters>(m, "InvalidParameters");

}
