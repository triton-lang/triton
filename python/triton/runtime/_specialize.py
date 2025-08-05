import functools
from triton.runtime.build import compile_module_from_src
from triton.backends.nvidia.driver import library_dirs, include_dirs

ARG_SPECIALIZE_SRC = """
#include <Python.h>
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>
// Pre-allocated common objects to avoid repeated allocation
static PyObject* str_i32 = NULL;
static PyObject* str_i64 = NULL;
static PyObject* str_u64 = NULL;
static PyObject* str_constexpr = NULL;
static PyObject* str_D = NULL;
static PyObject* str_empty = NULL;
static PyObject* tuple_constexpr_1 = NULL;
static PyObject* int_1 = NULL;
// Initialize common objects
static int init_common_objects(void) {
    str_i32 = PyUnicode_InternFromString("i32");
    str_i64 = PyUnicode_InternFromString("i64");
    str_u64 = PyUnicode_InternFromString("u64");
    str_constexpr = PyUnicode_InternFromString("constexpr");
    str_D = PyUnicode_InternFromString("D");
    str_empty = PyUnicode_InternFromString("");
    int_1 = PyLong_FromLong(1);
    // Pre-build the constexpr tuple for value=1
    tuple_constexpr_1 = PyTuple_New(2);

    Py_INCREF(str_constexpr);
    Py_INCREF(int_1);
    PyTuple_SET_ITEM(tuple_constexpr_1, 0, str_constexpr);
    PyTuple_SET_ITEM(tuple_constexpr_1, 1, int_1);

    return 0;
}
// integer specialization with type determination and alignment check
static PyObject* specialize_int(PyObject* self, PyObject* args) {
    // expects (value, specialize_value, align)
    PyObject* value_obj = PyTuple_GET_ITEM(args, 0);
    PyObject* specialize_value_obj = PyTuple_GET_ITEM(args, 1);
    PyObject* align_obj = PyTuple_GET_ITEM(args, 2);

    int specialize_value = PyObject_IsTrue(specialize_value_obj);
    int align = PyObject_IsTrue(align_obj);

    int overflow_i32 = 0;
    int overflow_i64 = 0;
    long val = 0;
    val = PyLong_AsLong(value_obj);
    overflow_i64 = PyErr_Occurred() ? 1 : 0;
    PyErr_Clear();
    overflow_i32 = overflow_i64 || (val < INT_MIN) || (val > INT_MAX);
    if (val == 1 && !overflow_i32 && specialize_value) {
        Py_INCREF(tuple_constexpr_1);
        return tuple_constexpr_1;
    }

    PyObject* type_str = NULL;
    if (!overflow_i32) {
        type_str = str_i32;
    } else {
        if (!overflow_i64) {
            type_str = str_i64;
        } else {
            type_str = str_u64;
        }
    }

    PyObject* result = PyTuple_New(2);
    if (!result) return NULL;
    Py_INCREF(type_str);
    PyTuple_SET_ITEM(result, 0, type_str);

    if (!specialize_value) {
        Py_INCREF(Py_None);
        PyTuple_SET_ITEM(result, 1, Py_None);
    } else if (align) {
        if (!overflow_i64) {
            if (val % 16 == 0) {
                Py_INCREF(str_D);
                PyTuple_SET_ITEM(result, 1, str_D);
            } else {
                Py_INCREF(str_empty);
                PyTuple_SET_ITEM(result, 1, str_empty);
            }
        } else {
            unsigned long long val_u64 = PyLong_AsUnsignedLongLong(value_obj);
            if (val_u64 % 16 == 0) {
                Py_INCREF(str_D);
                PyTuple_SET_ITEM(result, 1, str_D);
            } else {
                Py_INCREF(str_empty);
                PyTuple_SET_ITEM(result, 1, str_empty);
            }
        }
    } else {
        Py_INCREF(str_empty);
        PyTuple_SET_ITEM(result, 1, str_empty);
    }

    return result;
}
static PyObject* specialize_tensor(PyObject* self, PyObject* args) {
    // expects (data_ptr, align)
    PyObject* data_ptr_obj = PyTuple_GET_ITEM(args, 0);
    PyObject* align_obj = PyTuple_GET_ITEM(args, 1);
    uint64_t data_ptr = PyLong_AsUnsignedLongLong(data_ptr_obj);
    int align = PyLong_AsLong(align_obj);
    if (align && (data_ptr % 16 == 0)) {
        Py_INCREF(str_D);
        return str_D;
    } else {
        Py_INCREF(str_empty);
        return str_empty;
    }
}
static PyMethodDef SpecializeMethods[] = {
    {"specialize_int", specialize_int, METH_VARARGS, "integer specialization with type determination"},
    {"specialize_tensor", specialize_tensor, METH_VARARGS, "tensor data pointer alignment check"},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef SpecializeModule = {
    PyModuleDef_HEAD_INIT,
    "__triton_specialize",
    "Fast alignment checks and type determination for Triton specialization",
    -1,
    SpecializeMethods
};
PyMODINIT_FUNC PyInit___triton_specialize(void) {
    PyObject* module = PyModule_Create(&SpecializeModule);
    if (!module) {
        return NULL;
    }

    // Initialize common objects for performance
    if (init_common_objects() < 0) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
"""


class ArgSpecializer:

    def __init__(self, specialize_extra):
        mod = compile_module_from_src(
            src=ARG_SPECIALIZE_SRC,
            name="__triton_specialize",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=[],
        )

        use_fallback = "HIP" in specialize_extra.__qualname__

        def _specialize_int_fallback(arg, specialize_value, align):
            key = specialize_extra(arg, "int", align=align) if specialize_value else None
            if arg == 1 and specialize_value:
                return ("constexpr", 1)
            elif -(2**31) <= arg and arg <= 2**31 - 1:
                return ("i32", key)
            elif 2**63 <= arg and arg <= 2**64 - 1:
                return ("u64", key)
            else:
                return ("i64", key)

        def _specialize_tensor_fallback(arg, specialize_value, align):
            return specialize_extra(arg, "tensor", align=align) if specialize_value else None

        def _specialize_tensor(arg, specialize_value, align):
            return mod.specialize_tensor(arg.data_ptr(), specialize_value, align)

        if use_fallback:
            self.specialize_int = _specialize_int_fallback
            self.specialize_tensor = _specialize_tensor_fallback
        else:
            self.specialize_int = mod.specialize_int
            self.specialize_tensor = _specialize_tensor
