#pragma once

#include <Python.h>

#include "AllocatorCommon.h"

#include <limits>

namespace triton::cuda_allocator {

inline bool parseIntArg(PyObject *obj, const char *name, int *out) {
  long value = PyLong_AsLong(obj);
  if (value == -1 && PyErr_Occurred())
    return false;
  if (value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    PyErr_Format(PyExc_OverflowError, "%s is out of range for int", name);
    return false;
  }
  *out = static_cast<int>(value);
  return true;
}

inline bool parseShareableHandleTypeArg(PyObject *obj, const char *name,
                                        CUmemAllocationHandleType *out) {
  int handleType = 0;
  if (!parseIntArg(obj, name, &handleType))
    return false;
  if (handleType != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
      handleType != CU_MEM_HANDLE_TYPE_FABRIC) {
    PyErr_Format(PyExc_ValueError, "%s has unsupported value %d", name,
                 handleType);
    return false;
  }
  *out = static_cast<CUmemAllocationHandleType>(handleType);
  return true;
}

inline bool parseShareableHandleArg(PyObject *obj, const char *name,
                                    CUmemAllocationHandleType handleType,
                                    ShareableHandle *out) {
  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    return parseIntArg(obj, name, &out->fd);

  if (!PyBytes_Check(obj)) {
    PyErr_Format(PyExc_TypeError, "%s must be bytes", name);
    return false;
  }

  char *data = nullptr;
  Py_ssize_t size = 0;
  if (PyBytes_AsStringAndSize(obj, &data, &size) != 0)
    return false;
  if (size != static_cast<Py_ssize_t>(sizeof(out->fabricHandle))) {
    PyErr_Format(PyExc_ValueError, "%s must contain exactly %zu bytes, got %zd",
                 name, sizeof(out->fabricHandle), size);
    return false;
  }

  memcpy(&out->fabricHandle, data, sizeof(out->fabricHandle));
  return true;
}

inline PyObject *
shareableHandleToPyObject(const ShareableHandle &handle,
                          CUmemAllocationHandleType handleType) {
  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    return PyLong_FromLong(handle.fd);
  return PyBytes_FromStringAndSize(
      reinterpret_cast<const char *>(&handle.fabricHandle),
      sizeof(handle.fabricHandle));
}

inline bool parseVoidPtrArg(PyObject *obj, void **out) {
  *out = PyLong_AsVoidPtr(obj);
  return !(*out == nullptr && PyErr_Occurred());
}

} // namespace triton::cuda_allocator
