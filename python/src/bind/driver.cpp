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
#include "isaac/driver/backend.h"
#include "isaac/driver/buffer.h"
#include "isaac/driver/error.h"
#include "isaac/driver/cublas.h"
#include "isaac/driver/stream.h"
#include "isaac/driver/context.h"
#include "isaac/driver/platform.h"
#include "isaac/driver/module.h"
#include "isaac/driver/kernel.h"

namespace py = pybind11;
namespace drv = isaac::driver;
using be = drv::backend;

void export_driver(py::module&& m)
{
  //backend
  m.def("init", &be::init, "manually initialize the driver");
  m.def("release", &be::release, "manually release the driver");
  m.def("platforms", &be::platforms, "query available platforms");
  m.def("synchronize", &be::synchronize, "synchronize all detected devices");
  m.def("default_context", &be::contexts::get_default, "get default context");
  m.def("default_stream", &be::streams::get_default, "get default stream");

  //libraries
  m.def("cublasGemm", &drv::cublasGemm, "Wrapper for xGEMM");
  m.def("cudnnConv", &drv::cudnnConv, "Wrapper for xCONV");

  //backend state
  m.attr("default_device") = py::cast(&be::default_device);

  //frontend
  py::class_<drv::Platform>(m, "Platform")
      .def_property_readonly("name", &drv::Platform::name)
      .def_property_readonly("version", &drv::Platform::version)
      .def_property_readonly("devices", &drv::Platform::devices);

  py::class_<drv::Device>(m, "Device")
      .def_property_readonly("name", &drv::Device::name)
      .def_property_readonly("platform", &drv::Device::platform)
      .def_property_readonly("compute_capability", &drv::Device::compute_capability)
      .def_property_readonly("max_shared_memory", &drv::Device::max_shared_memory)
      .def_property_readonly("max_block_dim", &drv::Device::max_block_dim)
      .def_property_readonly("max_threads_per_block", &drv::Device::max_threads_per_block)
      .def_property_readonly("current_sm_clock", &drv::Device::current_sm_clock)
      .def_property_readonly("max_sm_clock", &drv::Device::max_sm_clock)
      .def_property_readonly("current_mem_clock", &drv::Device::current_mem_clock)
      .def_property_readonly("max_mem_clock", &drv::Device::max_mem_clock);

  py::class_<drv::Context>(m, "Context")
      .def(py::init<drv::Device>())
      .def_property_readonly("device", (drv::Device const & (drv::Context::*)() const)&drv::Context::device);

  py::class_<drv::Module>(m, "Module")
      .def(py::init<drv::Context, std::string>())
      .def_property_readonly("context", &drv::Module::context);

  py::class_<drv::Kernel>(m, "Kernel")
      .def(py::init<drv::Module, const char*>());

  py::class_<drv::Buffer>(m, "Buffer")
      .def(py::init<drv::Context, size_t>());

  py::class_<drv::Stream>(m, "Stream")
      .def(py::init<drv::Context>())
      .def("synchronize", &drv::Stream::synchronize);
      

  py::register_exception<drv::exception::cuda::base>(m, "CudaException");
  py::register_exception<drv::exception::cuda::launch_out_of_resources>(m, "CudaLaunchOutOfResources");
  py::register_exception<drv::exception::cuda::misaligned_address>(m, "CudaMisalignedAddress");
  py::register_exception<drv::exception::cuda::illegal_address>(m, "CudaIllegalAddress");

}
