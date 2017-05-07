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
#include "isaac/scalar.h"
namespace py = pybind11;
namespace sc = isaac;


void export_common(py::module& m)
{
  py::enum_<sc::DType>(m, "dtype")
      .value("float16", sc::DType::HALF_TYPE)
      .value("float32", sc::DType::FLOAT_TYPE)
      .value("float64", sc::DType::DOUBLE_TYPE)
      .export_values();

  m.def("size_of", sc::size_of);

  py::class_<sc::scalar>(m, "Scalar")
      .def(py::init<float, sc::DType>())
      .def(py::init<double, sc::DType>())
      .def_property_readonly("dtype", &sc::scalar::dtype);
}
