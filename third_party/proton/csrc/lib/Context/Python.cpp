#include "Context/Python.h"
#include "pybind11/pybind11.h"
#include <string>

namespace proton {

namespace {

std::string UnpackPyobject(PyObject *pyObject) {
  if (PyBytes_Check(pyObject)) {
    size_t size = PyBytes_GET_SIZE(pyObject);
    return std::string(PyBytes_AS_STRING(pyObject), size);
  }
  if (PyUnicode_Check(pyObject)) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t size;
    const char *data = PyUnicode_AsUTF8AndSize(pyObject, &size);
    if (!data) {
      return "";
    }
    return std::string(data, (size_t)size);
  }
  return "";
}

} // namespace

std::vector<Context> PythonContextSource::getContexts() {
  pybind11::gil_scoped_acquire gil;

  PyFrameObject *frame = PyEval_GetFrame();
  Py_XINCREF(frame);

  std::vector<Context> contexts;
  while (frame != nullptr) {
    PyCodeObject *f_code = PyFrame_GetCode(frame);
    size_t lineno = PyFrame_GetLineNumber(frame);
    size_t firstLineNo = f_code->co_firstlineno;
    std::string file = UnpackPyobject(f_code->co_filename);
    std::string function = UnpackPyobject(f_code->co_name);
    auto pythonFrame = file + ":" + function + "@" + std::to_string(lineno);
    contexts.push_back(Context(pythonFrame));
    auto newFrame = PyFrame_GetBack(frame);
    Py_DECREF(frame);
    frame = newFrame;
  }
  return contexts;
}

} // namespace proton
