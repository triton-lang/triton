#include <pybind11/pybind11.h>

void init_superblocking(pybind11::module &m);
void init_torch_utils(pybind11::module &m);
void init_triton(pybind11::module &m);
void init_cutlass(pybind11::module &m);

PYBIND11_MODULE(libtriton, m) {
  m.doc() = "Python bindings to the C++ Triton API";
  init_triton(m);
}
