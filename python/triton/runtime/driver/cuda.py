import hashlib
import os
import tempfile

from ...common.build import _build
from ..cache import get_cache_manager


def get_cuda_utils():
    global _cuda_utils
    if _cuda_utils is None:
        _cuda_utils = CudaUtils()
    return _cuda_utils


_cuda_utils = None


class CudaUtils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def _generate_src():
        return """
        #include <cuda.h>

        #include \"cuda.h\"
        #define PY_SSIZE_T_CLEAN
        #include <Python.h>

        static inline void gpuAssert(CUresult code, const char *file, int line)
        {
           if (code != CUDA_SUCCESS)
           {
              const char* prefix = "Triton Error [CUDA]: ";
              const char* str;
              cuGetErrorString(code, &str);
              char err[1024] = {0};
              strcat(err, prefix);
              strcat(err, str);
              PyErr_SetString(PyExc_RuntimeError, err);
           }
        }

        #define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); if(PyErr_Occurred()) return NULL; }

        static PyObject* getDeviceProperties(PyObject* self, PyObject* args){
            int device_id;
            if(!PyArg_ParseTuple(args, "i", &device_id))
                return NULL;
            // Get device handle
            CUdevice device;
            cuDeviceGet(&device, device_id);

            // create a struct to hold device properties
            int max_shared_mem;
            int multiprocessor_count;
            int sm_clock_rate;
            int mem_clock_rate;
            int mem_bus_width;
            CUDA_CHECK(cuDeviceGetAttribute(&max_shared_mem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device));
            CUDA_CHECK(cuDeviceGetAttribute(&multiprocessor_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
            CUDA_CHECK(cuDeviceGetAttribute(&sm_clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
            CUDA_CHECK(cuDeviceGetAttribute(&mem_clock_rate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
            CUDA_CHECK(cuDeviceGetAttribute(&mem_bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));


            return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i}", "max_shared_mem", max_shared_mem,
                                       "multiprocessor_count", multiprocessor_count,
                                       "sm_clock_rate", sm_clock_rate,
                                       "mem_clock_rate", mem_clock_rate,
                                       "mem_bus_width", mem_bus_width);
        }

        static PyObject* loadBinary(PyObject* self, PyObject* args) {
            const char* name;
            const char* data;
            Py_ssize_t data_size;
            int shared;
            int device;
            if(!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared, &device)) {
                return NULL;
            }
            CUfunction fun;
            CUmodule mod;
            int32_t n_regs = 0;
            int32_t n_spills = 0;
            // create driver handles
            CUDA_CHECK(cuModuleLoadData(&mod, data));
            CUDA_CHECK(cuModuleGetFunction(&fun, mod, name));
            // get allocated registers and spilled registers from the function
            CUDA_CHECK(cuFuncGetAttribute(&n_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fun));
            CUDA_CHECK(cuFuncGetAttribute(&n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
            n_spills /= 4;
            // set dynamic shared memory if necessary
            int shared_optin;
            CUDA_CHECK(cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device));
            if (shared > 49152 && shared_optin > 49152) {
              CUDA_CHECK(cuFuncSetCacheConfig(fun, CU_FUNC_CACHE_PREFER_SHARED));
              int shared_total, shared_static;
              CUDA_CHECK(cuDeviceGetAttribute(&shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device));
              CUDA_CHECK(cuFuncGetAttribute(&shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun));
              CUDA_CHECK(cuFuncSetAttribute(fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin - shared_static));
            }

            if(PyErr_Occurred()) {
              return NULL;
            }
            return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs, n_spills);
        }

        static PyMethodDef ModuleMethods[] = {
          {"load_binary", loadBinary, METH_VARARGS, "Load provided cubin into CUDA driver"},
          {"get_device_properties", getDeviceProperties, METH_VARARGS, "Get the properties for a given device"},
          {NULL, NULL, 0, NULL} // sentinel
        };

        static struct PyModuleDef ModuleDef = {
          PyModuleDef_HEAD_INIT,
          \"cuda_utils\",
          NULL, //documentation
          -1, //size
          ModuleMethods
        };

        PyMODINIT_FUNC PyInit_cuda_utils(void) {
          PyObject *m = PyModule_Create(&ModuleDef);
          if(m == NULL) {
            return NULL;
          }
          PyModule_AddFunctions(m, ModuleMethods);
          return m;
        }
        """

    def __init__(self):
        src = self._generate_src()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "cuda_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build("cuda_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("cuda_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
