#include <pybind11/pybind11.h>
namespace py = pybind11;

void init_triton_env_vars(pybind11::module &m);
void init_triton_ir(pybind11::module &&m);
void init_triton_interpreter(pybind11::module &&m);
void init_triton_passes(pybind11::module &&m);
void init_triton_translation(py::module &&m);

PYBIND11_MODULE(libtriton, m) {
  m.doc() = "Python bindings to the C++ Triton API";
  init_triton_env_vars(m);
  init_triton_ir(m.def_submodule("ir"));
  init_triton_passes(m.def_submodule("passes"));
  init_triton_interpreter(m.def_submodule("interpreter"));
  init_triton_translation(m.def_submodule("translation"));
}
