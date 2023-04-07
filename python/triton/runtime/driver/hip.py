import hashlib
import os
import tempfile

from ...common.build import _build
from ..cache import get_cache_manager


def get_hip_utils():
    global _hip_utils
    if _hip_utils is None:
        _hip_utils = HIPUtils()
    return _hip_utils


_hip_utils = None


class HIPUtils(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(HIPUtils, cls).__new__(cls)
        return cls.instance

    def _generate_src(self):
        return """
        #define __HIP_PLATFORM_AMD__
        #include <hip/hip_runtime.h>
        #define PY_SSIZE_T_CLEAN
        #include <Python.h>
        #include <stdio.h>
        #include <stdlib.h>
        static inline void gpuAssert(hipError_t code, const char *file, int line)
        {{
          if (code != HIP_SUCCESS)
          {{
             const char* prefix = "Triton Error [HIP]: ";
             const char* str = hipGetErrorString(code);
             char err[1024] = {0};
             snprintf(err, 1024, "%s Code: %d, Messsage: %s", prefix, code, str );
             PyErr_SetString(PyExc_RuntimeError, err);
          }}
        }}

        #define HIP_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); if (PyErr_Occurred()) return NULL; }

        static PyObject* getDeviceProperties(PyObject* self, PyObject* args){
            int device_id;
            if (!PyArg_ParseTuple(args, "i", &device_id))
                return NULL;

            hipDeviceProp_t props;
            HIP_CHECK(hipGetDeviceProperties(&props, device_id));

            // create a struct to hold device properties
            return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i}", "max_shared_mem", props.sharedMemPerBlock,
                                       "multiprocessor_count", props.multiProcessorCount,
                                       "sm_clock_rate", props.clockRate,
                                       "mem_clock_rate", props.memoryClockRate,
                                       "mem_bus_width", props.memoryBusWidth);
        }

        static PyObject* loadBinary(PyObject* self, PyObject* args) {
            const char* name;
            const char* data;
            Py_ssize_t data_size;
            int shared;
            int device;
            if (!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared, &device)) {
                return NULL;
            }

            // Open HSACO file
            FILE* hsaco_file;
            if ((hsaco_file = fopen(data, "rb")) == NULL) {
                return NULL;
            }

            // Read HSCAO file into Buffer
            fseek(hsaco_file, 0L, SEEK_END);
            size_t hsaco_file_size = ftell(hsaco_file);
            unsigned char* hsaco = (unsigned char*) malloc(hsaco_file_size * sizeof(unsigned char));
            rewind(hsaco_file);
            fread(hsaco, sizeof(unsigned char), hsaco_file_size, hsaco_file);
            fclose(hsaco_file);

            // set HIP options
            hipJitOption opt[] = {hipJitOptionErrorLogBufferSizeBytes, hipJitOptionErrorLogBuffer,
                                  hipJitOptionInfoLogBufferSizeBytes, hipJitOptionInfoLogBuffer,
                                  hipJitOptionLogVerbose};
            const unsigned int errbufsize = 8192;
            const unsigned int logbufsize = 8192;
            char _err[errbufsize];
            char _log[logbufsize];
            void *optval[] = {(void *)(uintptr_t)errbufsize,
                              (void *)_err, (void *)(uintptr_t)logbufsize,
                              (void *)_log, (void *)1};

            // launch HIP Binary
            hipModule_t mod;
            hipFunction_t fun;
            hipModuleLoadDataEx(&mod, hsaco, 5, opt, optval);
            hipModuleGetFunction(&fun, mod, name);
            free(hsaco);

            // get allocated registers and spilled registers from the function
            int n_regs = 0;
            int n_spills = 0;
            if (PyErr_Occurred()) {
              return NULL;
            }
            return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs, n_spills);
        }

        static PyMethodDef ModuleMethods[] = {
          {"load_binary", loadBinary, METH_VARARGS, "Load provided hsaco into HIP driver"},
          {"get_device_properties", getDeviceProperties, METH_VARARGS, "Get the properties for a given device"},
          {NULL, NULL, 0, NULL} // sentinel
        };

        static struct PyModuleDef ModuleDef = {
          PyModuleDef_HEAD_INIT,
          "hip_utils",
          NULL, //documentation
          -1, //size
          ModuleMethods
        };

        PyMODINIT_FUNC PyInit_hip_utils(void) {
          PyObject *m = PyModule_Create(&ModuleDef);
          if (m == NULL) {
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
        fname = "hip_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build("hip_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("hip_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
