#define _GNU_SOURCE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <link.h>
#include <stdbool.h>
#include <string.h>

#define MAX_PATH_LENGTH 4096

struct FindLibraryData {
  const char *lib_name;
  char path[MAX_PATH_LENGTH + 1];
  bool found;
  bool path_too_long;
};

// dl_iterate_phdr invokes this callback while holding the dynamic linker lock.
// Keep it entirely native: calling the Python API here would require the GIL
// and could deadlock with another GIL-holding thread waiting on the linker.
static int findLibraryCallback(struct dl_phdr_info *info, size_t size,
                               void *data) {
  (void)size;
  struct FindLibraryData *find_data = data;
  const char *path = info->dlpi_name;
  if (path == NULL || path[0] == '\0')
    return 0;

  const char *basename = strrchr(path, '/');
  basename = basename == NULL ? path : basename + 1;
  if (strstr(basename, find_data->lib_name) == NULL)
    return 0;

  size_t path_length = strnlen(path, MAX_PATH_LENGTH + 1);
  if (path_length > MAX_PATH_LENGTH) {
    find_data->path_too_long = true;
    return 1;
  }

  memcpy(find_data->path, path, path_length + 1);
  find_data->found = true;
  return 1;
}

static PyObject *findLoadedLibrary(PyObject *self, PyObject *args) {
  (void)self;
  const char *lib_name;
  if (!PyArg_ParseTuple(args, "s", &lib_name))
    return NULL;

  struct FindLibraryData data = {
      .lib_name = lib_name,
      .path = {0},
      .found = false,
      .path_too_long = false,
  };

  Py_BEGIN_ALLOW_THREADS
  dl_iterate_phdr(findLibraryCallback, &data);
  Py_END_ALLOW_THREADS

  if (data.path_too_long) {
    PyErr_SetString(PyExc_RuntimeError, "loaded library path exceeds 4096 bytes");
    return NULL;
  }
  if (!data.found)
    Py_RETURN_NONE;
  return PyUnicode_DecodeFSDefault(data.path);
}

static PyMethodDef ModuleMethods[] = {
    {"find_loaded_library", findLoadedLibrary, METH_VARARGS,
     "Return the path of the first loaded library matching the name."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "amd_dl_helper",
    .m_size = -1,
    .m_methods = ModuleMethods,
};

PyMODINIT_FUNC PyInit_amd_dl_helper(void) {
  return PyModule_Create(&ModuleDef);
}
