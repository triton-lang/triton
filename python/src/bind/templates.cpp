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
#include "isaac/templates/pool.h"
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

  py::enum_<sc::ActivationType>(m, "activation")
      .value("LINEAR", sc::ActivationType::Linear)
      .value("RELU", sc::ActivationType::ReLU)
      .value("ELU", sc::ActivationType::ELU)
      .value("SIGMOID", sc::ActivationType::Sigmoid)
      .export_values();

  py::enum_<sc::ResidualType>(m, "residual")
      .value("NO_RESIDUAL", sc::ResidualType::NoResidual)
      .value("ADD_RESIDUAL", sc::ResidualType::AddResidual)
      .value("CAT_RESIDUAL", sc::ResidualType::CatResidual)
      .export_values();

  py::enum_<sc::PoolType>(m, "pool")
      .value("MAX_POOL", sc::PoolType::MaxPool)
      .value("AVG_POOL", sc::PoolType::AvgPool)
      .export_values();

  py::class_<tpt::GEMM>(m, "GEMM")
      .def(py::init<sc::DType,sc::DType,sc::IsaacOperation_t,sc::IsaacOperation_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,
                    param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t>())
      .def("dump", &tpt::GEMM::dump)
      .def("enqueue", &tpt::GEMM::enqueue)
      .def_static("check_valid", check_valid<tpt::GEMM>)
      .def_readonly_static("id", &tpt::GEMM::id)
      .def_readonly_static("Nshapes", &tpt::GEMM::Nshapes)
      .def_readonly_static("Ntune", &tpt::GEMM::Ntune)
      .def_readonly_static("Nparams", &tpt::GEMM::Nparams);



  py::class_<tpt::Conv>(m, "Conv")
      .def(py::init<
                 sc::DType, sc::DType,
                  param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, //shapes
                  param_t, param_t, param_t, //pad
                  param_t, param_t, param_t, //stride
                  param_t, param_t, param_t, //upsample
                  sc::ActivationType, param_t, //number of outputs
                  sc::ResidualType, param_t, param_t, param_t, param_t, param_t, param_t, param_t, // Residual
                  param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t // Tuning
                 >())
      .def("dump", &tpt::Conv::dump)
      .def("enqueue", &tpt::Conv::enqueue)
      .def_static("check_valid", check_valid<tpt::Conv>)
      .def_readonly_static("id", &tpt::Conv::id)
      .def_readonly_static("Nshapes", &tpt::Conv::Nshapes)
      .def_readonly_static("Ntune", &tpt::Conv::Ntune)
      .def_readonly_static("Nparams", &tpt::Conv::Nparams);


  py::class_<tpt::Pool>(m, "Pool")
      .def(py::init<sc::DType, sc::DType, sc::PoolType, param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t,
                    param_t,param_t,param_t,param_t,param_t,param_t,param_t,param_t>())
      .def("dump", &tpt::Pool::dump)
      .def("enqueue", &tpt::Pool::enqueue)
      .def_static("check_valid", check_valid<tpt::Pool>)
      .def_readonly_static("id", &tpt::Pool::id)
      .def_readonly_static("Nshapes", &tpt::Pool::Nshapes)
      .def_readonly_static("Ntune", &tpt::Pool::Ntune)
      .def_readonly_static("Nparams", &tpt::Pool::Nparams);

  py::register_exception<tpt::invalid_parameters>(m, "InvalidParameters");

}
