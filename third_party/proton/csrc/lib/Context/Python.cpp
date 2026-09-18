#include "Context/Python.h"
#include "Utility/String.h"
#include <nanobind/nanobind.h>
#include <string>
#include <utility>

namespace proton {

namespace {

PyObject *asPyObject(PyFrameObject *frame) {
  return reinterpret_cast<PyObject *>(frame);
}

PyObject *asPyObject(PyCodeObject *code) {
  return reinterpret_cast<PyObject *>(code);
}

// bpo-42262 added Py_NewRef() to Python 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3 && !defined(Py_NewRef)
PyObject *_Py_NewRef(PyObject *obj) {
  Py_INCREF(obj);
  return obj;
}
#define Py_NewRef(obj) _Py_NewRef((PyObject *)(obj))
#endif

// bpo-42262 added Py_XNewRef() to Python 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3 && !defined(Py_XNewRef)
PyObject *_Py_XNewRef(PyObject *obj) {
  Py_XINCREF(obj);
  return obj;
}
#define Py_XNewRef(obj) _Py_XNewRef((PyObject *)(obj))
#endif

PyCodeObject *getFrameCodeObject(PyFrameObject *frame) {
  assert(frame != nullptr);
  return PyFrame_GetCode(frame);
}

PyFrameObject *getFrameBack(PyFrameObject *frame) {
  assert(frame != nullptr);
  PyObject *back = PyObject_GetAttrString(asPyObject(frame), "f_back");
  if (!back) {
    PyErr_Clear();
    return nullptr;
  }
  if (back == Py_None) {
    Py_DECREF(back);
    return nullptr;
  }
  return reinterpret_cast<PyFrameObject *>(back);
}

std::string unpackPyobject(PyObject *pyObject) {
  if (PyBytes_Check(pyObject)) {
    Py_ssize_t size = PyBytes_Size(pyObject);
    if (size < 0) {
      return "";
    }
    char *data = PyBytes_AsString(pyObject);
    if (!data) {
      return "";
    }
    return std::string(data, (size_t)size);
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

std::string unpackCodeAttr(PyCodeObject *code, const char *name) {
  PyObject *attr = PyObject_GetAttrString(asPyObject(code), name);
  if (!attr) {
    PyErr_Clear();
    return "";
  }
  std::string result = unpackPyobject(attr);
  Py_DECREF(attr);
  return result;
}

} // namespace

std::vector<Context> PythonContextSource::getContextsImpl() {
  nanobind::gil_scoped_acquire gil;

  PyFrameObject *frame = PyEval_GetFrame();
  Py_XINCREF(asPyObject(frame));

  std::vector<Context> reversedContexts;
  while (frame != nullptr) {
    PyCodeObject *f_code = getFrameCodeObject(frame);
    if (!f_code) {
      Py_DECREF(asPyObject(frame));
      break;
    }
    size_t lineno = PyFrame_GetLineNumber(frame);
    std::string file = unpackCodeAttr(f_code, "co_filename");
    std::string function = unpackCodeAttr(f_code, "co_name");
    Py_DECREF(asPyObject(f_code));
    auto pythonFrame = formatFileLineFunction(file, lineno, function);
    reversedContexts.emplace_back(std::move(pythonFrame));
    auto newFrame = getFrameBack(frame);
    Py_DECREF(asPyObject(frame));
    frame = newFrame;
  }
  std::vector<Context> contexts;
  contexts.reserve(reversedContexts.size());
  for (auto iter = reversedContexts.rbegin(); iter != reversedContexts.rend();
       ++iter) {
    contexts.push_back(*iter);
  }
  return contexts;
}

size_t PythonContextSource::getDepth() { return getContextsImpl().size(); }

} // namespace proton
