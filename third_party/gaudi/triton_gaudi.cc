// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>

namespace py = nanobind;

// Gaudi lowering is intentionally Python-first in the initial backend.  The
// native registration point is still required by Triton's backend plugin ABI;
// future Gaudi dialect/pass bindings are added to this module without changing
// the Python package or cache ABI.
void init_triton_gaudi(py::module_ &m) {
  m.attr("backend_name") = "gaudi";
  m.attr("artifact_abi") = 1;
}
