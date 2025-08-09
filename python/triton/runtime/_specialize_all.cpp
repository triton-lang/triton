#include <Python.h>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

using DTypePtrKey = std::pair<PyObject *, bool>;
using DTypeKey = PyObject *;
struct DTypePtrKeyHash {
  std::size_t operator()(const DTypePtrKey &k) const {
    return std::hash<PyObject *>()(k.first) ^
           (std::hash<bool>()(k.second) << 1);
  }
};
struct DTypeKeyHash {
  std::size_t operator()(const DTypeKey &k) const {
    return std::hash<PyObject *>()(k);
  }
};
using DtypePtr2Str =
    std::unordered_map<DTypePtrKey, PyObject *, DTypePtrKeyHash>;
using Dtype2Str = std::unordered_map<DTypeKey, PyObject *, DTypeKeyHash>;

static PyObject *constexpr_cls;
static PyObject *JITFunction;
static PyObject *TensorDescriptor;
static PyObject *GluonTensorDescriptor;
static PyObject *canonicalize_dtype;
static PyObject *canonicalize_ptr_dtype;

static PyObject *i32_str;
static PyObject *i64_str;
static PyObject *u64_str;
static PyObject *f32_str;
static PyObject *u1_str;
static PyObject *D_str;
static PyObject *constexpr_str;
static PyObject *empty_str;
static PyObject *nvTmaDesc_str;

static PyObject *data_ptr_attr;
static PyObject *dtype_attr;
static PyObject *cache_key_attr;
static PyObject *tma_desc_cpu_ptr_attr;
static PyObject *_fields_attr;
static PyObject *block_shape_attr;
static PyObject *layout_attr;

static DtypePtr2Str dtype_ptr2str;
static Dtype2Str dtype2str;

static inline bool is_tensor(PyObject *obj) {
  PyTypeObject *obj_type = Py_TYPE(obj);
  const char *type_name = obj_type->tp_name;
  return type_name && strcmp(type_name, "Tensor") == 0;
}

static inline std::pair<PyObject *, PyObject *>
_specialize_int(PyObject *arg, bool specialize_value, bool align) {
  PyObject *type_str;
  PyObject *key = Py_None;

  int overflow;
  long val = PyLong_AsLongAndOverflow(arg, &overflow);

  if (overflow == 0) {
    if (val >= INT32_MIN && val <= INT32_MAX) {
      type_str = i32_str;
    } else {
      type_str = i64_str;
    }

    if (specialize_value) {
      if (align) {
        key = (val & 15 == 0) ? D_str : empty_str; // % 16
      } else {
        key = empty_str;
      }
    }
  } else {
    type_str = u64_str;
    if (specialize_value) {
      unsigned long long val_u64 = PyLong_AsUnsignedLongLong(arg);
      key = (align && (val_u64 & 15 == 0)) ? D_str : empty_str; // % 16
    }
  }
  Py_INCREF(type_str);
  Py_INCREF(key);
  return {type_str, key};
}

static inline PyObject *
_specialize_tensor_align(PyObject *tensor, bool specialize_value, bool align) {
  if (!specialize_value) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  PyObject *data_ptr_method = PyObject_GetAttr(tensor, data_ptr_attr);

  PyObject *data_ptr_result = PyObject_CallNoArgs(data_ptr_method);
  Py_DECREF(data_ptr_method);

  uint64_t data_ptr = PyLong_AsUnsignedLongLong(data_ptr_result);
  Py_DECREF(data_ptr_result);

  PyObject *result;
  if (align && (data_ptr % 16 == 0)) {
    result = D_str;
  } else {
    result = empty_str;
  }

  Py_INCREF(result);
  return result;
}

static inline std::pair<PyObject *, PyObject *>
_specialize_tensor(PyObject *arg, bool is_const, bool specialize_value,
                   bool align) {
  PyObject *type_str;
  PyObject *key;
  PyObject *dtype = PyObject_GetAttr(arg, dtype_attr);
  DTypePtrKey dsk{arg, is_const};
  auto it = dtype_ptr2str.find(dsk);

  if (it != dtype_ptr2str.end()) {
    type_str = it->second;
    Py_INCREF(type_str);
  } else {
    PyObject *result = PyObject_CallFunctionObjArgs(
        canonicalize_ptr_dtype, dtype, is_const ? Py_True : Py_False, nullptr);
    dtype_ptr2str[dsk] = result;
    type_str = result;
    Py_INCREF(type_str);
  }

  Py_DECREF(dtype);
  key = _specialize_tensor_align(arg, specialize_value, align);

  return {type_str, key};
}

static inline std::pair<PyObject *, PyObject *>
_specialize_tensordesc(PyObject *arg, bool has_layout) {
  std::string result = "tensordesc<";

  PyObject *dtype = PyObject_GetAttr(arg, dtype_attr);
  DTypeKey dsk{arg};
  auto it = dtype2str.find(dsk);
  PyObject *type_str;
  if (it != dtype2str.end()) {
    type_str = it->second;
  } else {
    PyObject *result =
        PyObject_CallFunctionObjArgs(canonicalize_dtype, dtype, nullptr);
    dtype2str[dsk] = result;
    type_str = result;
  }

  Py_DECREF(dtype);
  PyObject *dtype_str = PyObject_Str(type_str);

  const char *dtype_cstr = PyUnicode_AsUTF8(dtype_str);
  result += dtype_cstr;
  Py_DECREF(dtype_str);

  PyObject *block_shape_obj = PyObject_GetAttr(arg, block_shape_attr);
  PyObject *block_shape_list = PySequence_List(block_shape_obj);
  Py_DECREF(block_shape_obj);
  PyObject *block_shape_str = PyObject_Str(block_shape_list);
  Py_DECREF(block_shape_list);

  const char *block_shape_cstr = PyUnicode_AsUTF8(block_shape_str);
  result += block_shape_cstr;
  Py_DECREF(block_shape_str);

  if (has_layout) {
    PyObject *layout_obj = PyObject_GetAttr(arg, layout_attr);
    PyObject *layout_repr = PyObject_Repr(layout_obj);
    Py_DECREF(layout_obj);

    const char *layout_cstr = PyUnicode_AsUTF8(layout_repr);
    result += ",";
    result += layout_cstr;
    Py_DECREF(layout_repr);
  }

  result += ">";

  PyObject *type_str_result = PyUnicode_FromString(result.c_str());
  Py_INCREF(Py_None);
  return {type_str_result, Py_None};
}

std::pair<PyObject *, PyObject *> _specialize_arg(PyObject *arg, bool is_const,
                                                  bool specialize_value,
                                                  bool align) {
  if (is_tensor(arg)) {
    return _specialize_tensor(arg, is_const, specialize_value, align);
  }

  PyTypeObject *arg_type = Py_TYPE(arg);
  if (arg_type == &PyLong_Type) {
    return _specialize_int(arg, specialize_value, align);
  }

  if (arg == Py_None) {
    Py_INCREF(constexpr_str);
    Py_INCREF(Py_None);
    return {constexpr_str, Py_None};
  }

  if (arg_type == &PyBool_Type) {
    Py_INCREF(u1_str);
    Py_INCREF(Py_None);
    return {u1_str, Py_None};
  }

  if (arg_type == &PyFloat_Type) {
    Py_INCREF(f32_str);
    Py_INCREF(Py_None);
    return {f32_str, Py_None};
  }

  if (PyObject_IsInstance(arg, constexpr_cls)) {
    Py_INCREF(constexpr_str);
    Py_INCREF(arg);
    return {constexpr_str, arg};
  }

  if (PyObject_HasAttr(arg, data_ptr_attr)) {
    return _specialize_tensor(arg, is_const, specialize_value, align);
  }

  if (PyObject_IsInstance(arg, JITFunction)) {
    Py_INCREF(constexpr_str);
    return {constexpr_str, PyObject_GetAttr(arg, cache_key_attr)};
  }

  if (PyObject_HasAttr(arg, tma_desc_cpu_ptr_attr)) {
    Py_INCREF(nvTmaDesc_str);
    Py_INCREF(Py_None);
    return {nvTmaDesc_str, Py_None};
  }

  if (PyTuple_Check(arg)) {
    Py_ssize_t tuple_size = PyTuple_Size(arg);

    PyObject **tys = (PyObject **)malloc(tuple_size * sizeof(PyObject *));
    PyObject **keys = (PyObject **)malloc(tuple_size * sizeof(PyObject *));

    for (Py_ssize_t i = 0; i < tuple_size; ++i) {
      auto [ty, key] = _specialize_arg(PyTuple_GetItem(arg, i), is_const,
                                       specialize_value, align);
      tys[i] = ty;
      keys[i] = key;
    }

    PyObject *out_tys;
    PyObject *out_keys;

    if (PyObject_HasAttr(arg, _fields_attr)) {
      PyObject *tuple_type = (PyObject *)Py_TYPE(arg);

      PyObject *tys_tuple = PyTuple_New(tuple_size);
      PyObject *keys_tuple = PyTuple_New(tuple_size);
      for (Py_ssize_t i = 0; i < tuple_size; ++i) {
        PyTuple_SET_ITEM(tys_tuple, i, tys[i]);
        PyTuple_SET_ITEM(keys_tuple, i, keys[i]);
      }

      out_tys = PyObject_CallFunctionObjArgs(tuple_type, tys_tuple, nullptr);
      out_keys = PyObject_CallFunctionObjArgs(tuple_type, keys_tuple, nullptr);

      Py_DECREF(tys_tuple);
      Py_DECREF(keys_tuple);
    } else {
      // Regular tuple
      out_tys = PyTuple_New(tuple_size);
      out_keys = PyTuple_New(tuple_size);
      for (Py_ssize_t i = 0; i < tuple_size; ++i) {
        PyTuple_SET_ITEM(out_tys, i, tys[i]);
        PyTuple_SET_ITEM(out_keys, i, keys[i]);
      }
    }

    free(tys);
    free(keys);
    return {out_tys, out_keys};
  }

  if (PyObject_IsInstance(arg, TensorDescriptor)) {
    return _specialize_tensordesc(arg, false);
  }

  if (PyObject_IsInstance(arg, GluonTensorDescriptor)) {
    return _specialize_tensordesc(arg, true);
  }

  // TODO: throw error
  Py_INCREF(Py_None);
  return {Py_None, Py_None};
}

static PyObject *specialize_impl(PyObject *self, PyObject *args) {
  PyObject *arg_obj;
  int is_const_int, specialize_value_int, align_int;

  arg_obj = PyTuple_GetItem(args, 0);

  PyObject *const_obj = PyTuple_GetItem(args, 1);
  PyObject *spec_obj = PyTuple_GetItem(args, 2);
  PyObject *align_obj = PyTuple_GetItem(args, 3);

  bool is_const = PyObject_IsTrue(const_obj);
  bool specialize_value = PyObject_IsTrue(spec_obj);
  bool align = PyObject_IsTrue(align_obj);

  auto [type, key] =
      _specialize_arg(arg_obj, is_const, specialize_value, align);

  PyObject *result_tuple = PyTuple_New(2);
  PyTuple_SET_ITEM(result_tuple, 0, type);
  PyTuple_SET_ITEM(result_tuple, 1, key);

  return result_tuple;
}

static PyMethodDef module_methods[] = {
    {"specialize_impl", specialize_impl, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT,
                                        "__triton_specialize_all", nullptr, -1,
                                        module_methods};

PyMODINIT_FUNC PyInit___triton_specialize_all(void) {
  PyObject *module = PyModule_Create(&module_def);
  if (!module)
    return nullptr;

  i32_str = PyUnicode_InternFromString("i32");
  i64_str = PyUnicode_InternFromString("i64");
  u64_str = PyUnicode_InternFromString("u64");
  f32_str = PyUnicode_InternFromString("f32");
  u1_str = PyUnicode_InternFromString("u1");
  D_str = PyUnicode_InternFromString("D");
  constexpr_str = PyUnicode_InternFromString("constexpr");
  empty_str = PyUnicode_InternFromString("");
  nvTmaDesc_str = PyUnicode_InternFromString("nvTmaDesc");

  data_ptr_attr = PyUnicode_InternFromString("data_ptr");
  dtype_attr = PyUnicode_InternFromString("dtype");
  cache_key_attr = PyUnicode_InternFromString("cache_key");
  tma_desc_cpu_ptr_attr = PyUnicode_InternFromString("tma_desc_cpu_ptr");
  _fields_attr = PyUnicode_InternFromString("_fields");
  block_shape_attr = PyUnicode_InternFromString("block_shape");
  layout_attr = PyUnicode_InternFromString("layout");

  PyObject *m_jit = PyImport_ImportModule("triton.runtime.jit");
  if (!m_jit)
    return nullptr;
  JITFunction = PyObject_GetAttrString(m_jit, "JITFunction");

  PyObject *m_desc = PyImport_ImportModule("triton.tools.tensor_descriptor");
  if (!m_desc)
    return nullptr;
  TensorDescriptor = PyObject_GetAttrString(m_desc, "TensorDescriptor");

  PyObject *m_desc_gluon =
      PyImport_ImportModule("triton.experimental.gluon.nvidia.hopper");
  if (!m_desc_gluon)
    return nullptr;
  GluonTensorDescriptor =
      PyObject_GetAttrString(m_desc_gluon, "TensorDescriptor");

  PyObject *m_canonicalize = PyImport_ImportModule("triton._utils");
  if (!m_canonicalize)
    return nullptr;
  canonicalize_dtype =
      PyObject_GetAttrString(m_canonicalize, "canonicalize_dtype");
  canonicalize_ptr_dtype =
      PyObject_GetAttrString(m_canonicalize, "canonicalize_ptr_dtype");

  PyObject *m_constexpr = PyImport_ImportModule("triton.language");
  if (!m_constexpr)
    return nullptr;
  constexpr_cls = PyObject_GetAttrString(m_constexpr, "constexpr");

  Py_DECREF(m_jit);
  Py_DECREF(m_desc);
  Py_DECREF(m_desc_gluon);
  Py_DECREF(m_canonicalize);
  Py_DECREF(m_constexpr);

  return module;
}
