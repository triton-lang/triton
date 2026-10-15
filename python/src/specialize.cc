#include <Python.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <nanobind/nanobind.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace py = nanobind;

using DTypePtrKey = std::pair<Py_hash_t, bool>;
using DTypeKey = Py_hash_t;

struct DTypePtrKeyHash {
  std::size_t operator()(const DTypePtrKey &k) const {
    return std::hash<Py_hash_t>()(k.first) ^ (std::hash<bool>()(k.second) << 1);
  }
};

using DtypePtr2Str =
    std::unordered_map<DTypePtrKey, PyObject *, DTypePtrKeyHash>;
using Dtype2Str = std::unordered_map<DTypeKey, PyObject *>;

using TypeHandler = std::pair<py::object, py::object> (*)(PyObject *,
                                                          PyObject *, bool,
                                                          bool, bool);
using TypeHandlerCache = std::unordered_map<PyTypeObject *, TypeHandler>;

static std::pair<py::object, py::object>
specialize_arg(PyObject *backend, PyObject *arg, bool is_const,
               bool specialize_value, bool align);

static bool init_called = false;

static PyObject *constexpr_cls = nullptr;
static PyObject *jit_callable_cls = nullptr;
static PyObject *tensor_descriptor_cls = nullptr;
static PyObject *nvidia_tensor_descriptor_cls = nullptr;
static PyObject *nvidia_tensor_descriptor_im2col_cls = nullptr;
static PyObject *amd_tensor_descriptor_cls = nullptr;
static PyObject *canonicalize_dtype_fn = nullptr;
static PyObject *canonicalize_ptr_dtype_fn = nullptr;
static PyObject *torch_tensor_cls = nullptr;

static PyObject *i32_str = nullptr;
static PyObject *i64_str = nullptr;
static PyObject *u64_str = nullptr;
static PyObject *fp32_str = nullptr;
static PyObject *u1_str = nullptr;
static PyObject *D_str = nullptr;
static PyObject *constexpr_str = nullptr;
static PyObject *empty_str = nullptr;
static PyObject *nvTmaDesc_str = nullptr;

static PyObject *base_attr = nullptr;
static PyObject *data_ptr_attr = nullptr;
static PyObject *dtype_attr = nullptr;
static PyObject *cache_key_attr = nullptr;
static PyObject *_fields_attr = nullptr;
static PyObject *block_shape_attr = nullptr;
static PyObject *shape_attr = nullptr;
static PyObject *layout_attr = nullptr;
static PyObject *has_native_tensor_spec_attr = nullptr;
static PyObject *get_tensor_spec_attr = nullptr;
static PyObject *align_kwarg = nullptr;
static PyObject *debug_str = nullptr;
static PyObject *instrumentation_mode_str = nullptr;
static PyObject *fpsan_homomorphic_casts_str = nullptr;

static DtypePtr2Str dtype_ptr2str;
static Dtype2Str dtype2str;
static TypeHandlerCache type_handler_cache;

struct JITParamMeta {
  bool is_constexpr;
  bool is_const;
  bool specialize;
  bool align;
  PyObject *annotation_type;
};

// Cache-hit launcher for the common JIT path. It deliberately reuses the
// Python-owned cache dictionaries and falls back for unsupported calls.
typedef struct {
  PyObject_HEAD vectorcallfunc vectorcall;
  PyObject *jit_fn;
  PyObject *fallback;
  PyObject *backend;
  PyObject *kernel_cache;
  PyObject *kernel_key_cache;
  PyObject *options_str;
  PyObject *grid[3];
  PyObject *stream_getter;
  PyObject *device_getter;
  PyObject *device;
  PyObject *used_global_vals;
  PyObject *pre_run_hooks;
  PyObject *launch_enter_hooks;
  PyObject *launch_exit_hooks;
  PyObject *param_name_to_idx;
  PyObject *default_options[3];
  JITParamMeta *params;
  Py_ssize_t num_params;
} JITCacheProxy;

static PyTypeObject JITCacheProxyType = {PyVarObject_HEAD_INIT(nullptr, 0)};

// Wrappers to make steal and borrow slightly simpler. We use raw CPython API
// with py::object to handle decref, as higher-level binding APIs add exception
// handling overhead which is quite significant here.
py::object from_new_ref(py::handle val) { return py::steal<py::object>(val); }
py::object from_borrowed_ref(py::handle val) {
  return py::borrow<py::object>(val);
}

const char *unicode_as_utf8(PyObject *obj) {
  Py_ssize_t size;
  return PyUnicode_AsUTF8AndSize(obj, &size);
}

void set_specialize_type_error(PyObject *arg) {
  auto arg_type = from_new_ref(PyObject_Type(arg));
  if (!arg_type) {
    PyErr_SetString(PyExc_TypeError, "failed to specialize argument");
    return;
  }

  auto type_name =
      from_new_ref(PyObject_GetAttrString(arg_type.ptr(), "__name__"));
  if (!type_name) {
    PyErr_Clear();
    PyErr_SetString(PyExc_TypeError, "failed to specialize argument");
    return;
  }

  PyErr_Format(PyExc_TypeError, "failed to specialize argument of type: %U",
               type_name.ptr());
}

PyObject *intern_from_string(const char *str) {
  PyObject *obj = PyUnicode_InternFromString(str);
  if (!obj)
    throw py::python_error();
  return obj;
}

PyObject *import_from(const char *module_name, const char *var_name) {
  py::object var = py::module_::import_(module_name).attr(var_name);
  return var.release().ptr();
}

void init_interned_strings() {
  i32_str = intern_from_string("i32");
  i64_str = intern_from_string("i64");
  u64_str = intern_from_string("u64");
  fp32_str = intern_from_string("fp32");
  u1_str = intern_from_string("u1");
  D_str = intern_from_string("D");
  constexpr_str = intern_from_string("constexpr");
  empty_str = intern_from_string("");
  nvTmaDesc_str = intern_from_string("nvTmaDesc");

  base_attr = intern_from_string("base");
  data_ptr_attr = intern_from_string("data_ptr");
  dtype_attr = intern_from_string("dtype");
  cache_key_attr = intern_from_string("cache_key");
  _fields_attr = intern_from_string("_fields");
  block_shape_attr = intern_from_string("block_shape");
  shape_attr = intern_from_string("shape");
  layout_attr = intern_from_string("layout");
  has_native_tensor_spec_attr =
      intern_from_string("supports_native_tensor_specialization");
  get_tensor_spec_attr = intern_from_string("get_tensor_specialization");

  align_kwarg = py::make_tuple("align").release().ptr();
  debug_str = intern_from_string("debug");
  instrumentation_mode_str = intern_from_string("instrumentation_mode");
  fpsan_homomorphic_casts_str = intern_from_string("fpsan_homomorphic_casts");
}

void init_type_handler_cache();

bool init_globals() noexcept try {
  // Import releavant symbols
  jit_callable_cls = import_from("triton.runtime.jit", "JITCallable");
  tensor_descriptor_cls =
      import_from("triton.tools.tensor_descriptor", "TensorDescriptor");
  nvidia_tensor_descriptor_cls = import_from(
      "triton.experimental.gluon.nvidia.hopper", "TensorDescriptor");
  nvidia_tensor_descriptor_im2col_cls = import_from(
      "triton.experimental.gluon.nvidia.hopper", "TensorDescriptorIm2Col");
  amd_tensor_descriptor_cls =
      import_from("triton.experimental.gluon.amd.cdna5", "TensorDescriptor");

  auto m_canonicalize = py::module_::import_("triton._utils");
  canonicalize_dtype_fn = import_from("triton._utils", "canonicalize_dtype");
  canonicalize_ptr_dtype_fn =
      import_from("triton._utils", "canonicalize_ptr_dtype");
  constexpr_cls = import_from("triton.language", "constexpr");

  PyObject *loaded_modules = PyImport_GetModuleDict();
  PyObject *torch_module = PyDict_GetItemString(loaded_modules, "torch");
  if (torch_module) {
    auto tensor_cls =
        from_new_ref(PyObject_GetAttrString(torch_module, "Tensor"));
    if (!tensor_cls)
      return false;
    torch_tensor_cls = tensor_cls.release().ptr();
  }

  init_interned_strings();
  init_type_handler_cache();

  init_called = true;
  return true;
} catch (py::python_error &e) {
  e.restore();
  return false;
}

std::pair<py::object, py::object> specialize_tensordesc(PyObject *arg,
                                                        bool has_layout) {
  auto base = from_new_ref(PyObject_GetAttr(arg, base_attr));
  if (!base)
    return {};

  auto dtype = from_new_ref(PyObject_GetAttr(base.ptr(), dtype_attr));
  if (!dtype)
    return {};

  PyObject *type_str;
  Py_hash_t dtype_hash = PyObject_Hash(dtype.ptr());
  if (dtype_hash == -1)
    return {};
  DTypeKey dsk{dtype_hash};
  auto it = dtype2str.find(dsk);
  if (it != dtype2str.end()) {
    type_str = it->second;
  } else {
    auto res = from_new_ref(PyObject_CallFunctionObjArgs(canonicalize_dtype_fn,
                                                         dtype.ptr(), nullptr));
    if (!res)
      return {};
    dtype2str[dsk] = res.ptr();
    type_str = res.release().ptr();
  }

  std::string desc_cstr;
  desc_cstr.reserve(128);

  // Determine im2col by class type (Gluon only).
  bool is_im2col = false;
  if (has_layout && nvidia_tensor_descriptor_im2col_cls) {
    int is_inst = PyObject_IsInstance(arg, nvidia_tensor_descriptor_im2col_cls);
    if (is_inst < 0)
      return {};
    is_im2col = is_inst == 1;
  }

  desc_cstr = is_im2col ? "tensordesc_im2col<" : "tensordesc<";
  auto dtype_str = from_new_ref(PyObject_Str(type_str));
  if (!dtype_str)
    return {};

  const char *dtype_cstr = unicode_as_utf8(dtype_str.ptr());
  if (!dtype_cstr)
    return {};
  desc_cstr += dtype_cstr;

  auto block_shape_obj = from_new_ref(PyObject_GetAttr(arg, block_shape_attr));
  if (!block_shape_obj)
    return {};
  auto block_shape_list = from_new_ref(PySequence_List(block_shape_obj.ptr()));
  if (!block_shape_list)
    return {};
  auto block_shape_str = from_new_ref(PyObject_Str(block_shape_list.ptr()));
  if (!block_shape_str)
    return {};
  const char *block_shape_cstr = unicode_as_utf8(block_shape_str.ptr());
  if (!block_shape_cstr)
    return {};
  desc_cstr += block_shape_cstr;

  // For im2col mode, append input tensor rank after block_shape
  // Format: tensordesc_im2col<dtype[block_shape],input_rank=N,layout>
  // This allows the driver to know the N-dimensional shape/strides to pass
  if (is_im2col) {
    auto tensor_shape_obj = from_new_ref(PyObject_GetAttr(arg, shape_attr));
    if (!tensor_shape_obj)
      return {};
    Py_ssize_t tensor_rank = PySequence_Size(tensor_shape_obj.ptr());
    if (tensor_rank < 0)
      return {};
    desc_cstr += ",input_rank=";
    desc_cstr += std::to_string(tensor_rank);
  }

  if (has_layout) {
    auto layout_obj = from_new_ref(PyObject_GetAttr(arg, layout_attr));
    if (!layout_obj)
      return {};
    auto layout_repr = from_new_ref(PyObject_Repr(layout_obj.ptr()));
    if (!layout_repr)
      return {};
    desc_cstr += ",";
    const char *layout_cstr = unicode_as_utf8(layout_repr.ptr());
    if (!layout_cstr)
      return {};
    desc_cstr += layout_cstr;
  }

  desc_cstr += ">";
  auto type_str_result = from_new_ref(PyUnicode_FromString(desc_cstr.c_str()));
  if (!type_str_result)
    return {};

  return {std::move(type_str_result), py::none()};
}

std::pair<py::object, py::object> handle_long_type(PyObject *backend,
                                                   PyObject *arg, bool is_const,
                                                   bool specialize_value,
                                                   bool align) {
  int overflow;
  long long val = PyLong_AsLongLongAndOverflow(arg, &overflow);
  if (PyErr_Occurred()) {
    return {};
  }

  if (specialize_value && (val == 1)) {
    return {from_borrowed_ref(constexpr_str), from_borrowed_ref(arg)};
  }

  py::handle type_str;
  py::handle key_obj;
  if (overflow == 0) {
    type_str = (val >= INT32_MIN && val <= INT32_MAX) ? i32_str : i64_str;
    if (specialize_value) {
      key_obj = (align && ((val & 15) == 0)) ? D_str : empty_str;
    }
  } else {
    unsigned long long val_64 = PyLong_AsUnsignedLongLong(arg);
    if (PyErr_Occurred()) {
      // this runs into an edge-case where the Python reference
      // returns i64 as type and alignment of the value despite
      // not being representable as such which at kernel launch later
      // will throw an OverflowError nevertheless, here we throw
      // OverflowError immediately
      PyErr_SetString(PyExc_OverflowError,
                      "integer to be specialized too large to represent");
      return {};
    }
    type_str = u64_str;
    if (specialize_value) {
      key_obj = (align && ((val_64 & 15) == 0)) ? D_str : empty_str;
    }
  }
  if (!key_obj) {
    return {from_borrowed_ref(type_str), py::none()};
  }
  return {from_borrowed_ref(type_str), from_borrowed_ref(key_obj)};
}

std::pair<py::object, py::object> handle_tensor(PyObject *backend,
                                                PyObject *arg, bool is_const,
                                                bool specialize_value,
                                                bool align) {
  // handle type_str specialization of a tensor
  auto dtype = from_new_ref(PyObject_GetAttr(arg, dtype_attr));
  if (!dtype)
    return {};

  Py_hash_t dtype_hash = PyObject_Hash(dtype.ptr());
  if (dtype_hash == -1)
    return {};

  DTypePtrKey dsk{dtype_hash, is_const};
  auto it = dtype_ptr2str.find(dsk);

  py::handle type_str;
  if (it != dtype_ptr2str.end()) {
    type_str = it->second;
  } else {
    auto canon_res =
        PyObject_CallFunctionObjArgs(canonicalize_ptr_dtype_fn, dtype.ptr(),
                                     is_const ? Py_True : Py_False, nullptr);
    if (!canon_res)
      return {};
    dtype_ptr2str[dsk] = canon_res;
    type_str = canon_res;
  }

  // handle alignment specialization of a tensor
  if (!specialize_value) {
    return {from_borrowed_ref(type_str), py::none()};
  }

  bool native_impl_available = false;
  auto native_spec_obj =
      from_new_ref(PyObject_GetAttr(backend, has_native_tensor_spec_attr));
  if (native_spec_obj) {
    native_impl_available = PyObject_IsTrue(native_spec_obj.ptr());
  } else {
    PyErr_Clear();
    // on error we fall back to native_impl_available = false gracefully
  }

  py::object key;
  if (native_impl_available) {
    auto data_ptr_result =
        from_new_ref(PyObject_CallMethodObjArgs(arg, data_ptr_attr, nullptr));
    if (!data_ptr_result)
      return {};

    auto data_ptr = PyLong_AsUnsignedLongLong(data_ptr_result.ptr());
    if (PyErr_Occurred())
      return {};

    auto key_obj = (align && ((data_ptr & 15) == 0)) ? D_str : empty_str;
    key = from_borrowed_ref(key_obj);
  } else {
    PyObject *args[3] = {backend, arg, align ? Py_True : Py_False};
    PyObject *kwnames = align_kwarg;
    key = from_new_ref(
        PyObject_VectorcallMethod(get_tensor_spec_attr, args, 2, kwnames));
    if (!key)
      return {};
  }

  return {from_borrowed_ref(type_str), std::move(key)};
}

std::pair<py::object, py::object> handle_bool_type(PyObject *backend,
                                                   PyObject *arg, bool is_const,
                                                   bool specialize_value,
                                                   bool align) {
  return {from_borrowed_ref(u1_str), py::none()};
}

std::pair<py::object, py::object>
handle_float_type(PyObject *backend, PyObject *arg, bool is_const,
                  bool specialize_value, bool align) {
  return {from_borrowed_ref(fp32_str), py::none()};
}

std::pair<py::object, py::object>
handle_tensor_descriptor(PyObject *backend, PyObject *arg, bool is_const,
                         bool specialize_value, bool align) {
  return specialize_tensordesc(arg, false);
}

std::pair<py::object, py::object>
handle_gluon_tensor_descriptor(PyObject *backend, PyObject *arg, bool is_const,
                               bool specialize_value, bool align) {
  return specialize_tensordesc(arg, true);
}

std::pair<py::object, py::object>
handle_constexpr_type(PyObject *backend, PyObject *arg, bool is_const,
                      bool specialize_value, bool align) {
  return {from_borrowed_ref(constexpr_str), from_borrowed_ref(arg)};
}

std::pair<py::object, py::object>
handle_jit_callable(PyObject *backend, PyObject *arg, bool is_const,
                    bool specialize_value, bool align) {
  auto cache_key = from_new_ref(PyObject_GetAttr(arg, cache_key_attr));
  if (!cache_key)
    return {};
  return {from_borrowed_ref(constexpr_str), std::move(cache_key)};
}

std::pair<py::object, py::object> handle_tuple(PyObject *backend, PyObject *arg,
                                               bool is_const,
                                               bool specialize_value,
                                               bool align) {
  Py_ssize_t size = PyTuple_Size(arg);
  if (size < 0)
    return {};
  if (size == 0) {
    // return tuple of empty tuples as in python reference
    return {from_borrowed_ref(arg), from_borrowed_ref(arg)};
  }

  bool is_namedtuple = PyObject_HasAttr(arg, _fields_attr);
  auto tuple_type = Py_TYPE(arg);

  // Create tuples directly instead of lists
  auto tys_tuple = from_new_ref(PyTuple_New(size));
  if (!tys_tuple)
    return {};

  auto keys_tuple = from_new_ref(PyTuple_New(size));
  if (!keys_tuple)
    return {};

  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject *item = PyTuple_GetItem(arg, i); // Borrowed reference
    if (!item)
      return {};
    // python reference calls specialize recursively with default arguments set
    // currently this is is_const=False, specialize_value=True, align=True
    auto [type, key] = specialize_arg(backend, item, false, true, true);
    if (!type || !key)
      return {};
    // Steals references on success.
    if (PyTuple_SetItem(tys_tuple.ptr(), i, type.release().ptr()) < 0)
      return {};
    if (PyTuple_SetItem(keys_tuple.ptr(), i, key.release().ptr()) < 0)
      return {};
  }

  if (is_namedtuple) {
    tys_tuple = from_new_ref(
        PyObject_CallObject((PyObject *)tuple_type, tys_tuple.ptr()));
    if (!tys_tuple)
      return {};
    keys_tuple = from_new_ref(
        PyObject_CallObject((PyObject *)tuple_type, keys_tuple.ptr()));
    if (!keys_tuple)
      return {};
  }

  return {std::move(tys_tuple), std::move(keys_tuple)};
}

// initialize type handler which returns specialize impelemntations based on
// type(arg)
void init_type_handler_cache() {
  // Python Types (int, bool, float, tuple)
  type_handler_cache[&PyLong_Type] = handle_long_type;
  type_handler_cache[&PyBool_Type] = handle_bool_type;
  type_handler_cache[&PyFloat_Type] = handle_float_type;
  type_handler_cache[&PyTuple_Type] = handle_tuple;

  // torch.Tensor
  if (torch_tensor_cls && PyType_Check(torch_tensor_cls)) {
    type_handler_cache[(PyTypeObject *)torch_tensor_cls] = handle_tensor;
  }
  // TensorDescriptor
  if (tensor_descriptor_cls && PyType_Check(tensor_descriptor_cls)) {
    type_handler_cache[(PyTypeObject *)tensor_descriptor_cls] =
        handle_tensor_descriptor;
  }
  // GluonTensorDescriptor
  if (nvidia_tensor_descriptor_cls &&
      PyType_Check(nvidia_tensor_descriptor_cls)) {
    type_handler_cache[(PyTypeObject *)nvidia_tensor_descriptor_cls] =
        handle_gluon_tensor_descriptor;
  }
  if (nvidia_tensor_descriptor_im2col_cls &&
      PyType_Check(nvidia_tensor_descriptor_im2col_cls)) {
    type_handler_cache[(PyTypeObject *)nvidia_tensor_descriptor_im2col_cls] =
        handle_gluon_tensor_descriptor;
  }
  if (amd_tensor_descriptor_cls && PyType_Check(amd_tensor_descriptor_cls)) {
    type_handler_cache[(PyTypeObject *)amd_tensor_descriptor_cls] =
        handle_gluon_tensor_descriptor;
  }
  // constexpr
  if (constexpr_cls && PyType_Check(constexpr_cls)) {
    type_handler_cache[(PyTypeObject *)constexpr_cls] = handle_constexpr_type;
  }
  // JITCallable
  if (jit_callable_cls && PyType_Check(jit_callable_cls)) {
    type_handler_cache[(PyTypeObject *)jit_callable_cls] = handle_jit_callable;
  }
}

// specialization logic without passing of objects from Python (to be called in
// specialize_impl only)
std::pair<py::object, py::object> specialize_arg(PyObject *backend,
                                                 PyObject *arg, bool is_const,
                                                 bool specialize_value,
                                                 bool align) {
  // fast-path for default types
  PyTypeObject *arg_type = Py_TYPE(arg);
  auto it = type_handler_cache.find(arg_type);
  if (it != type_handler_cache.end()) {
    return it->second(backend, arg, is_const, specialize_value, align);
  }

  // separate handling of None
  if (Py_IsNone(arg)) {
    return {from_borrowed_ref(constexpr_str), py::none()};
  }

  // handling of sublcasses of tuples
  if (PyTuple_Check(arg)) {
    return handle_tuple(backend, arg, is_const, specialize_value, align);
  }

  // fallback paths checking full inheritance
  if (PyObject_IsInstance(arg, constexpr_cls)) {
    return handle_constexpr_type(backend, arg, is_const, specialize_value,
                                 align);
  }

  if (PyObject_IsInstance(arg, tensor_descriptor_cls)) {
    return handle_tensor_descriptor(backend, arg, is_const, specialize_value,
                                    align);
  }

  if (PyObject_IsInstance(arg, nvidia_tensor_descriptor_cls)) {
    return handle_gluon_tensor_descriptor(backend, arg, is_const,
                                          specialize_value, align);
  }

  if (PyObject_IsInstance(arg, amd_tensor_descriptor_cls)) {
    return handle_gluon_tensor_descriptor(backend, arg, is_const,
                                          specialize_value, align);
  }

  if (PyObject_IsInstance(arg, jit_callable_cls)) {
    return handle_jit_callable(backend, arg, is_const, specialize_value, align);
  }

  // fallback paths checking attributes directly
  if (PyObject_HasAttr(arg, data_ptr_attr)) {
    return handle_tensor(backend, arg, is_const, specialize_value, align);
  }

  // fallback for default types
  if (PyLong_Check(arg)) {
    return handle_long_type(backend, arg, is_const, specialize_value, align);
  }
  if (PyFloat_Check(arg)) {
    return handle_float_type(backend, arg, is_const, specialize_value, align);
  }

  return {};
}

// main entry-point from Python implementing specialization logic natively
PyObject *specialize_impl(PyObject *self, PyObject *const *args,
                          Py_ssize_t nargs) {
  if (!init_called) {
    if (!init_globals()) {
      return nullptr;
    }
  }

  if (nargs != 5) {
    PyErr_SetString(PyExc_TypeError,
                    "native_specialize_impl expected 5 arguments");
    return nullptr;
  }

  PyObject *backend = args[0];
  PyObject *arg = args[1];
  int is_const = PyObject_IsTrue(args[2]);
  int specialize_value = PyObject_IsTrue(args[3]);
  int align = PyObject_IsTrue(args[4]);

  if (is_const == -1 || specialize_value == -1 || align == -1) {
    PyErr_SetString(PyExc_TypeError, "native_specialize_impl expected boolean "
                                     "arguments for args2, args3, args4");
    return nullptr;
  }

  auto [type, key] =
      specialize_arg(backend, arg, is_const, specialize_value, align);

  // check if specialization failed
  if (!type || !key) {
    if (!PyErr_Occurred())
      set_specialize_type_error(arg);
    return nullptr;
  }

  return PyTuple_Pack(2, type.ptr(), key.ptr());
}

bool visit_make_tensordesc_args(PyObject *arg, PyObject *sig,
                                PyObject *relevant_paths,
                                PyObject *tensordesc_meta,
                                bool has_tensordesc_meta, PyObject *base_args,
                                PyObject *make_tensordesc_arg,
                                Py_ssize_t *tensordesc_idx, PyObject *result) {
  assert(PyTuple_Check(sig));
  auto arg_fast =
      from_new_ref(PySequence_Fast(arg, "Expected iterable args node"));
  if (!arg_fast)
    return false;

  Py_ssize_t arg_len = PySequence_Size(arg_fast.ptr());
  if (arg_len < 0)
    return false;
  Py_ssize_t sig_len = PyTuple_Size(sig);
  if (sig_len < 0)
    return false;
  assert(sig_len == arg_len || !"Invalid signature");
  Py_ssize_t len = arg_len;

  for (Py_ssize_t i = 0; i < len; ++i) {
    auto a_obj = from_new_ref(PySequence_GetItem(arg_fast.ptr(), i));
    if (!a_obj)
      return false;
    PyObject *a = a_obj.ptr();
    PyObject *s = PyTuple_GetItem(sig, i);
    if (!s)
      return false;

    if (PyUnicode_CheckExact(s)) {
      Py_ssize_t size;
      const char *type_cstr = PyUnicode_AsUTF8AndSize(s, &size);
      if (!type_cstr)
        return false;

      // if not s.startswith("tensordesc")
      std::string_view tensordesc = "tensordesc";
      std::string_view type_str(type_cstr, size);
      if (type_str.substr(0, tensordesc.length()) != tensordesc) {
        if (PyList_Append(result, a) < 0)
          return false;
        continue;
      }

      PyObject *meta = Py_None;
      if (has_tensordesc_meta) {
        // Borrowed reference
        meta = PyList_GetItem(tensordesc_meta, *tensordesc_idx);
        if (!meta)
          return false;
      }

      PyObject *vector_args[] = {a, meta, base_args};
      auto desc_args = from_new_ref(
          PyObject_Vectorcall(make_tensordesc_arg, vector_args, 3, nullptr));
      if (!desc_args)
        return false;
      // list.extend(desc_args)
      if (PyList_SetSlice(result, PY_SSIZE_T_MAX, PY_SSIZE_T_MAX,
                          desc_args.ptr()) < 0)
        return false;

      *tensordesc_idx += 1;
      continue;
    }

    auto key = from_new_ref(PyLong_FromSsize_t(i));
    if (!key)
      return false;

    // Borrowed ref
    PyObject *inner_relevant_paths =
        PyDict_GetItemWithError(relevant_paths, key.ptr());
    if (PyErr_Occurred())
      return false;

    if (!inner_relevant_paths) {
      // Short-circuit if tuple doesn't contain any tensordesc args
      if (PyList_Append(result, a) < 0)
        return false;
      continue;
    }

    // Recurse into tuple
    auto inner_res = from_new_ref(PyList_New(0));
    if (!inner_res)
      return false;
    if (!visit_make_tensordesc_args(
            a, s, inner_relevant_paths, tensordesc_meta, has_tensordesc_meta,
            base_args, make_tensordesc_arg, tensordesc_idx, inner_res.ptr()))
      return false;

    auto inner_tuple = from_new_ref(PyList_AsTuple(inner_res.ptr()));
    if (!inner_tuple)
      return false;
    if (PyList_Append(result, inner_tuple.ptr()) < 0)
      return false;
  }
  return true;
}

PyObject *make_tensordesc_args(PyObject *self, PyObject *const *args,
                               Py_ssize_t nargs) {
  if (nargs != 6) {
    PyErr_SetString(PyExc_TypeError,
                    "make_tensordesc_args expected 6 arguments");
    return nullptr;
  }

  PyObject *kernel_args = args[0];
  PyObject *signature = args[1];
  PyObject *relevant_paths = args[2];
  PyObject *tensordesc_meta = args[3];
  PyObject *base_args = args[4];
  PyObject *make_tensordesc_arg = args[5];

  if (!PyList_CheckExact(tensordesc_meta)) {
    PyErr_SetString(PyExc_TypeError, "Expected tensordesc_meta to be a list");
    return nullptr;
  }
  Py_ssize_t tensordesc_meta_len = PyList_Size(tensordesc_meta);
  if (tensordesc_meta_len < 0)
    return nullptr;
  bool has_tensordesc_meta = tensordesc_meta_len > 0;

  auto result = from_new_ref(PyList_New(0));
  if (!result)
    return nullptr;

  Py_ssize_t tensordesc_idx = 0;
  if (!visit_make_tensordesc_args(kernel_args, signature, relevant_paths,
                                  tensordesc_meta, has_tensordesc_meta,
                                  base_args, make_tensordesc_arg,
                                  &tensordesc_idx, result.ptr()))
    return nullptr;

  if (has_tensordesc_meta) {
    Py_ssize_t meta_len = PySequence_Size(tensordesc_meta);
    if (meta_len < 0)
      return nullptr;

    if (tensordesc_idx != meta_len) {
      PyErr_SetString(PyExc_ValueError,
                      "make_tensordesc_args: tensordesc_idx != meta_len");
      return nullptr;
    }
  }

  return result.release().ptr();
}

py::object build_specialization(JITCacheProxy *proxy, PyObject *const *args) {
  auto specialization = from_new_ref(PyTuple_New(proxy->num_params));
  if (!specialization)
    return {};

  for (Py_ssize_t i = 0; i < proxy->num_params; ++i) {
    JITParamMeta &meta = proxy->params[i];
    PyObject *arg = args[i];
    py::object item;

    if (meta.is_constexpr) {
      item = from_new_ref(PyTuple_Pack(2, constexpr_str, arg));
    } else if (meta.annotation_type && !meta.specialize) {
      item = from_new_ref(PyTuple_Pack(2, meta.annotation_type, Py_None));
    } else {
      auto [type, key] = specialize_arg(proxy->backend, arg, meta.is_const,
                                        meta.specialize, meta.align);
      if (!type || !key)
        return {};
      item = from_new_ref(PyTuple_Pack(
          2, meta.annotation_type ? meta.annotation_type : type.ptr(),
          key.ptr()));
    }

    if (!item)
      return {};
    PyTuple_SET_ITEM(specialization.ptr(), i, item.release().ptr());
  }
  return specialization;
}

PyObject *JITCacheProxy_vectorcall(PyObject *callable, PyObject *const *args,
                                   size_t nargsf, PyObject *kwnames) {
  JITCacheProxy *proxy = reinterpret_cast<JITCacheProxy *>(callable);
  Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
  PyObject *const *effective_args = args;
  Py_ssize_t effective_nargs = nargs;
  std::vector<PyObject *> merged_args;
  std::vector<bool> filled;
  py::object dynamic_options_str;
  PyObject *effective_options_str = proxy->options_str;

  if (nargs > proxy->num_params)
    goto fallback;
  if (kwnames && PyTuple_GET_SIZE(kwnames) > 0) {
    merged_args.resize(proxy->num_params);
    filled.resize(proxy->num_params);
    for (Py_ssize_t i = 0; i < nargs; ++i) {
      merged_args[i] = args[i];
      filled[i] = true;
    }

    auto options = from_new_ref(PyDict_New());
    if (!options)
      return nullptr;
    Py_ssize_t num_kwargs = PyTuple_GET_SIZE(kwnames);
    for (Py_ssize_t i = 0; i < num_kwargs; ++i) {
      PyObject *name = PyTuple_GET_ITEM(kwnames, i);
      PyObject *value = args[nargs + i];
      PyObject *idx_obj =
          PyDict_GetItemWithError(proxy->param_name_to_idx, name);
      if (idx_obj) {
        Py_ssize_t idx = PyLong_AsSsize_t(idx_obj);
        if (idx < 0 || idx >= proxy->num_params || filled[idx]) {
          PyErr_Clear();
          goto fallback;
        }
        merged_args[idx] = value;
        filled[idx] = true;
      } else {
        if (PyErr_Occurred())
          return nullptr;
        if (PyDict_SetItem(options.ptr(), name, value) < 0)
          return nullptr;
      }
    }
    for (bool present : filled) {
      if (!present)
        goto fallback;
    }
    effective_args = merged_args.data();
    effective_nargs = proxy->num_params;

    if (PyDict_GET_SIZE(options.ptr()) > 0) {
      // Match JITFunction.run's injected options and insertion order so the
      // structural key is identical to compute_cache_key's Python key.
      PyObject *debug = PyDict_GetItemWithError(options.ptr(), debug_str);
      if (!debug && PyErr_Occurred())
        return nullptr;
      PyObject *effective_debug = proxy->default_options[0];
      if (debug) {
        int truth = PyObject_IsTrue(debug);
        if (truth < 0)
          return nullptr;
        if (truth)
          effective_debug = debug;
      }
      if (PyDict_SetItem(options.ptr(), debug_str, effective_debug) < 0 ||
          PyDict_SetItem(options.ptr(), instrumentation_mode_str,
                         proxy->default_options[1]) < 0 ||
          PyDict_SetItem(options.ptr(), fpsan_homomorphic_casts_str,
                         proxy->default_options[2]) < 0)
        return nullptr;
      dynamic_options_str = from_new_ref(PyObject_Str(options.ptr()));
      if (!dynamic_options_str)
        return nullptr;
      effective_options_str = dynamic_options_str.ptr();
    }
  } else if (nargs != proxy->num_params) {
    goto fallback;
  }
  if (PyDict_GET_SIZE(proxy->used_global_vals) != 0 ||
      PyList_GET_SIZE(proxy->pre_run_hooks) != 0 ||
      PyList_GET_SIZE(proxy->launch_enter_hooks) != 0 ||
      PyList_GET_SIZE(proxy->launch_exit_hooks) != 0)
    goto fallback;

  {
    auto current_device =
        from_new_ref(PyObject_CallNoArgs(proxy->device_getter));
    if (!current_device)
      return nullptr;
    int same_device =
        PyObject_RichCompareBool(current_device.ptr(), proxy->device, Py_EQ);
    if (same_device < 0)
      return nullptr;
    if (!same_device)
      goto fallback;

    auto specialization = build_specialization(proxy, effective_args);
    if (!specialization) {
      PyErr_Clear();
      goto fallback;
    }
    auto key = from_new_ref(
        PyTuple_Pack(2, specialization.ptr(), effective_options_str));
    if (!key)
      return nullptr;

    PyObject *cache_key =
        PyDict_GetItemWithError(proxy->kernel_key_cache, key.ptr());
    if (!cache_key) {
      if (PyErr_Occurred())
        PyErr_Clear();
      goto fallback;
    }
    PyObject *kernel = PyDict_GetItemWithError(proxy->kernel_cache, cache_key);
    if (!kernel) {
      if (PyErr_Occurred())
        PyErr_Clear();
      goto fallback;
    }

    auto launcher =
        from_new_ref(PyObject_GetAttrString(kernel, "_prebound_launcher"));
    if (!launcher) {
      PyErr_Clear();
      goto fallback;
    }
    if (launcher.ptr() == Py_None)
      goto fallback;

    auto stream = from_new_ref(
        PyObject_CallOneArg(proxy->stream_getter, current_device.ptr()));
    if (!stream)
      return nullptr;

    std::vector<PyObject *> launch_args(4 + effective_nargs);
    launch_args[0] = proxy->grid[0];
    launch_args[1] = proxy->grid[1];
    launch_args[2] = proxy->grid[2];
    launch_args[3] = stream.ptr();
    for (Py_ssize_t i = 0; i < effective_nargs; ++i)
      launch_args[4 + i] = effective_args[i];

    auto result = from_new_ref(PyObject_Vectorcall(
        launcher.ptr(), launch_args.data(), launch_args.size(), nullptr));
    if (!result)
      return nullptr;
    Py_INCREF(kernel);
    return kernel;
  }

fallback: {
  PyObject *result =
      PyObject_Vectorcall(proxy->fallback, args, nargsf, kwnames);
  if (!result)
    return nullptr;
  auto used_global_vals =
      from_new_ref(PyObject_GetAttrString(proxy->jit_fn, "used_global_vals"));
  if (!used_global_vals) {
    Py_DECREF(result);
    return nullptr;
  }
  Py_SETREF(proxy->used_global_vals, used_global_vals.release().ptr());
  return result;
}
}

void JITCacheProxy_dealloc(PyObject *obj) {
  PyObject_GC_UnTrack(obj);
  JITCacheProxy *proxy = reinterpret_cast<JITCacheProxy *>(obj);
  Py_CLEAR(proxy->jit_fn);
  Py_CLEAR(proxy->fallback);
  Py_CLEAR(proxy->backend);
  Py_CLEAR(proxy->kernel_cache);
  Py_CLEAR(proxy->kernel_key_cache);
  Py_CLEAR(proxy->options_str);
  for (PyObject *&value : proxy->grid)
    Py_CLEAR(value);
  Py_CLEAR(proxy->stream_getter);
  Py_CLEAR(proxy->device_getter);
  Py_CLEAR(proxy->device);
  Py_CLEAR(proxy->used_global_vals);
  Py_CLEAR(proxy->pre_run_hooks);
  Py_CLEAR(proxy->launch_enter_hooks);
  Py_CLEAR(proxy->launch_exit_hooks);
  Py_CLEAR(proxy->param_name_to_idx);
  for (PyObject *&value : proxy->default_options)
    Py_CLEAR(value);
  if (proxy->params) {
    for (Py_ssize_t i = 0; i < proxy->num_params; ++i)
      Py_XDECREF(proxy->params[i].annotation_type);
    PyMem_Free(proxy->params);
  }
  Py_TYPE(obj)->tp_free(obj);
}

int JITCacheProxy_traverse(PyObject *obj, visitproc visit, void *arg) {
  JITCacheProxy *proxy = reinterpret_cast<JITCacheProxy *>(obj);
  Py_VISIT(proxy->jit_fn);
  Py_VISIT(proxy->fallback);
  Py_VISIT(proxy->backend);
  Py_VISIT(proxy->kernel_cache);
  Py_VISIT(proxy->kernel_key_cache);
  Py_VISIT(proxy->options_str);
  for (PyObject *value : proxy->grid)
    Py_VISIT(value);
  Py_VISIT(proxy->stream_getter);
  Py_VISIT(proxy->device_getter);
  Py_VISIT(proxy->device);
  Py_VISIT(proxy->used_global_vals);
  Py_VISIT(proxy->pre_run_hooks);
  Py_VISIT(proxy->launch_enter_hooks);
  Py_VISIT(proxy->launch_exit_hooks);
  Py_VISIT(proxy->param_name_to_idx);
  for (PyObject *value : proxy->default_options)
    Py_VISIT(value);
  for (Py_ssize_t i = 0; i < proxy->num_params; ++i)
    Py_VISIT(proxy->params[i].annotation_type);
  return 0;
}

int JITCacheProxy_clear(PyObject *obj) {
  JITCacheProxy *proxy = reinterpret_cast<JITCacheProxy *>(obj);
  Py_CLEAR(proxy->jit_fn);
  Py_CLEAR(proxy->fallback);
  Py_CLEAR(proxy->backend);
  Py_CLEAR(proxy->kernel_cache);
  Py_CLEAR(proxy->kernel_key_cache);
  Py_CLEAR(proxy->options_str);
  for (PyObject *&value : proxy->grid)
    Py_CLEAR(value);
  Py_CLEAR(proxy->stream_getter);
  Py_CLEAR(proxy->device_getter);
  Py_CLEAR(proxy->device);
  Py_CLEAR(proxy->used_global_vals);
  Py_CLEAR(proxy->pre_run_hooks);
  Py_CLEAR(proxy->launch_enter_hooks);
  Py_CLEAR(proxy->launch_exit_hooks);
  Py_CLEAR(proxy->param_name_to_idx);
  for (PyObject *&value : proxy->default_options)
    Py_CLEAR(value);
  for (Py_ssize_t i = 0; i < proxy->num_params; ++i)
    Py_CLEAR(proxy->params[i].annotation_type);
  return 0;
}

bool get_bool_attr(PyObject *obj, const char *name, bool *value) {
  auto attr = from_new_ref(PyObject_GetAttrString(obj, name));
  if (!attr)
    return false;
  int truth = PyObject_IsTrue(attr.ptr());
  if (truth < 0)
    return false;
  *value = truth != 0;
  return true;
}

PyObject *create_jit_cache_proxy(PyObject *, PyObject *const *args,
                                 Py_ssize_t nargs) {
  constexpr Py_ssize_t expected_nargs = 16;
  if (nargs != expected_nargs) {
    PyErr_Format(PyExc_TypeError,
                 "native_create_jit_cache_proxy expected %zd arguments",
                 expected_nargs);
    return nullptr;
  }
  if (!init_called && !init_globals())
    return nullptr;

  PyObject *params = args[10];
  if (!PyList_CheckExact(params) || !PyDict_CheckExact(args[2]) ||
      !PyDict_CheckExact(args[3]) || !PyUnicode_CheckExact(args[4]) ||
      !PyTuple_CheckExact(args[5]) || !PyDict_CheckExact(args[9]) ||
      !PyList_CheckExact(args[11]) || !PyList_CheckExact(args[12]) ||
      !PyList_CheckExact(args[13]) || !PyTuple_CheckExact(args[15]) ||
      PyTuple_GET_SIZE(args[15]) != 3) {
    PyErr_SetString(PyExc_TypeError,
                    "invalid native JIT cache proxy arguments");
    return nullptr;
  }

  Py_ssize_t num_params = PyList_GET_SIZE(params);
  JITParamMeta *param_meta = static_cast<JITParamMeta *>(
      PyMem_Calloc(num_params, sizeof(JITParamMeta)));
  if (num_params && !param_meta)
    return PyErr_NoMemory();
  auto param_name_to_idx = from_new_ref(PyDict_New());
  if (!param_name_to_idx)
    goto error;

  for (Py_ssize_t i = 0; i < num_params; ++i) {
    PyObject *param = PyList_GET_ITEM(params, i);
    auto name = from_new_ref(PyObject_GetAttrString(param, "name"));
    auto idx = from_new_ref(PyLong_FromSsize_t(i));
    if (!name || !idx ||
        PyDict_SetItem(param_name_to_idx.ptr(), name.ptr(), idx.ptr()) < 0)
      goto error;
    bool do_not_specialize = false;
    bool do_not_specialize_on_alignment = false;
    if (!get_bool_attr(param, "is_constexpr", &param_meta[i].is_constexpr) ||
        !get_bool_attr(param, "is_const", &param_meta[i].is_const) ||
        !get_bool_attr(param, "do_not_specialize", &do_not_specialize) ||
        !get_bool_attr(param, "do_not_specialize_on_alignment",
                       &do_not_specialize_on_alignment)) {
      goto error;
    }
    param_meta[i].specialize = !do_not_specialize;
    param_meta[i].align = !do_not_specialize_on_alignment;

    auto annotation =
        from_new_ref(PyObject_GetAttrString(param, "annotation_type"));
    if (!annotation)
      goto error;
    if (PyUnicode_GET_LENGTH(annotation.ptr()) != 0) {
      const char *value = PyUnicode_AsUTF8(annotation.ptr());
      if (!value)
        goto error;
      if (std::strcmp(value, "u1") == 0 || std::strncmp(value, "fp", 2) == 0 ||
          std::strncmp(value, "bf", 2) == 0)
        param_meta[i].specialize = false;
      param_meta[i].annotation_type = annotation.release().ptr();
    }
  }

  {
    auto one = from_new_ref(PyLong_FromLong(1));
    if (!one)
      goto error;

    JITCacheProxy *proxy = reinterpret_cast<JITCacheProxy *>(
        PyObject_GC_New(JITCacheProxy, &JITCacheProxyType));
    if (!proxy)
      goto error;
    proxy->vectorcall = JITCacheProxy_vectorcall;
    proxy->params = param_meta;
    proxy->num_params = num_params;
    proxy->param_name_to_idx = param_name_to_idx.release().ptr();
    for (Py_ssize_t i = 0; i < 3; ++i) {
      proxy->default_options[i] = PyTuple_GET_ITEM(args[15], i);
      Py_INCREF(proxy->default_options[i]);
    }

    PyObject **fields[] = {
        &proxy->jit_fn,
        &proxy->fallback,
        &proxy->backend,
        &proxy->kernel_cache,
        &proxy->kernel_key_cache,
        &proxy->options_str,
        &proxy->stream_getter,
        &proxy->device_getter,
        &proxy->device,
        &proxy->used_global_vals,
        &proxy->pre_run_hooks,
        &proxy->launch_enter_hooks,
        &proxy->launch_exit_hooks,
    };
    int arg_indices[] = {14, 0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13};
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
      *fields[i] = args[arg_indices[i]];
      Py_INCREF(*fields[i]);
    }
    Py_ssize_t grid_size = PyTuple_GET_SIZE(args[5]);
    for (Py_ssize_t i = 0; i < 3; ++i) {
      proxy->grid[i] = i < grid_size ? PyTuple_GET_ITEM(args[5], i) : one.ptr();
      Py_INCREF(proxy->grid[i]);
    }
    PyObject_GC_Track(proxy);
    return reinterpret_cast<PyObject *>(proxy);
  }

error:
  for (Py_ssize_t i = 0; i < num_params; ++i)
    Py_XDECREF(param_meta[i].annotation_type);
  PyMem_Free(param_meta);
  return nullptr;
}

void init_jit_cache_proxy_type() {
  JITCacheProxyType.tp_name = "triton._C.libtriton._JITCacheProxy";
  JITCacheProxyType.tp_basicsize = sizeof(JITCacheProxy);
  JITCacheProxyType.tp_flags =
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL | Py_TPFLAGS_HAVE_GC;
  JITCacheProxyType.tp_vectorcall_offset = offsetof(JITCacheProxy, vectorcall);
  JITCacheProxyType.tp_call = PyVectorcall_Call;
  JITCacheProxyType.tp_dealloc = JITCacheProxy_dealloc;
  JITCacheProxyType.tp_traverse = JITCacheProxy_traverse;
  JITCacheProxyType.tp_clear = JITCacheProxy_clear;
}

static PyMethodDef module_methods[] = {
    {"native_specialize_impl", (PyCFunction)specialize_impl, METH_FASTCALL,
     nullptr},
    {"native_create_jit_cache_proxy", (PyCFunction)create_jit_cache_proxy,
     METH_FASTCALL, nullptr},
    {"make_tensordesc_args", (PyCFunction)make_tensordesc_args, METH_FASTCALL,
     "Helper to translate tensordesc args"},
    {nullptr, nullptr, 0, nullptr} // sentinel
};

} // anonymous namespace

void init_native_specialize(nanobind::module_ &m) {
  init_jit_cache_proxy_type();
  if (PyType_Ready(&JITCacheProxyType) < 0)
    throw nanobind::python_error();
  // add functions to module
  PyModule_AddFunctions(m.ptr(), module_methods);
}
