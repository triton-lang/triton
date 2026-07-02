/**
 * Metal runtime driver - Objective-C implementation.
 *
 * This file provides the native Metal API integration for kernel dispatch.
 * It is compiled as a Python extension module using the Metal framework.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <Python.h>

// Global Metal state
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;

static int metal_init(void) {
    if (g_device != nil) return 0;

    g_device = MTLCreateSystemDefaultDevice();
    if (g_device == nil) {
        return -1;
    }

    g_queue = [g_device newCommandQueue];
    if (g_queue == nil) {
        return -1;
    }

    return 0;
}

static PyObject* py_metal_init(PyObject* self, PyObject* args) {
    int result = metal_init();
    return PyLong_FromLong(result);
}

static PyObject* py_get_device_name(PyObject* self, PyObject* args) {
    if (metal_init() != 0) {
        Py_RETURN_NONE;
    }
    NSString* name = [g_device name];
    return PyUnicode_FromString([name UTF8String]);
}

static PyObject* py_get_gpu_family(PyObject* self, PyObject* args) {
    if (metal_init() != 0) {
        return PyLong_FromLong(0);
    }

    // Detect GPU family (Apple10 = M4, Apple9 = M3, Apple8 = M2, Apple7 = M1)
    if (@available(macOS 26.0, *)) {
        if ([g_device supportsFamily:MTLGPUFamilyApple10]) {
            return PyLong_FromLong(10);  // M4
        }
    }
    if ([g_device supportsFamily:MTLGPUFamilyApple9]) {
        return PyLong_FromLong(9);  // M3
    } else if ([g_device supportsFamily:MTLGPUFamilyApple8]) {
        return PyLong_FromLong(8);  // M2
    } else if ([g_device supportsFamily:MTLGPUFamilyApple7]) {
        return PyLong_FromLong(7);  // M1
    }

    return PyLong_FromLong(7);  // Default to Apple7
}

static PyObject* py_get_max_threadgroup_memory(PyObject* self, PyObject* args) {
    if (metal_init() != 0) {
        return PyLong_FromLong(0);
    }
    return PyLong_FromLong((long)[g_device maxThreadgroupMemoryLength]);
}

static PyObject* py_get_max_threads_per_threadgroup(PyObject* self, PyObject* args) {
    if (metal_init() != 0) {
        return PyLong_FromLong(0);
    }
    // Apple Silicon supports up to 1024 threads per threadgroup
    return PyLong_FromLong(1024);
}

static PyObject* py_load_binary(PyObject* self, PyObject* args) {
    const char* name;
    const char* binary_data;
    Py_ssize_t binary_len;
    int shared_memory;

    if (!PyArg_ParseTuple(args, "ss#i", &name, &binary_data, &binary_len, &shared_memory)) {
        return NULL;
    }

    if (metal_init() != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize Metal device");
        return NULL;
    }

    NSError* error = nil;
    NSData* data = [NSData dataWithBytes:binary_data length:binary_len];

    // Try loading as metallib first
    id<MTLLibrary> library = [g_device newLibraryWithData:dispatch_data_create(
        [data bytes], [data length], NULL, NULL) error:&error];

    if (library == nil) {
        // Fall back to JIT compilation from MSL source
        NSString* source = [[NSString alloc] initWithBytes:binary_data
                                                    length:binary_len
                                                  encoding:NSUTF8StringEncoding];
        if (source == nil) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to decode MSL source");
            return NULL;
        }

        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        if (@available(macOS 26.0, *)) {
            options.mathMode = MTLMathModeFast;
            options.languageVersion = MTLLanguageVersion4_0;
        } else if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeFast;
            options.languageVersion = MTLLanguageVersion3_2;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = YES;
#pragma clang diagnostic pop
            options.languageVersion = MTLLanguageVersion3_1;
        }

        library = [g_device newLibraryWithSource:source options:options error:&error];
        if (library == nil) {
            NSString* errMsg = [error localizedDescription];
            PyErr_SetString(PyExc_RuntimeError, [errMsg UTF8String]);
            return NULL;
        }
    }

    NSString* funcName = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library newFunctionWithName:funcName];
    if (function == nil) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to find kernel function");
        return NULL;
    }

    id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:function error:&error];
    if (pso == nil) {
        NSString* errMsg = [error localizedDescription];
        PyErr_SetString(PyExc_RuntimeError, [errMsg UTF8String]);
        return NULL;
    }

    // Return the pipeline state as a capsule
    return PyCapsule_New((__bridge_retained void*)pso, "metal_pso", NULL);
}

static PyObject* py_dispatch(PyObject* self, PyObject* args) {
    PyObject* pso_capsule;
    int grid_x, grid_y, grid_z;
    int threads_x, threads_y, threads_z;
    int shared_memory;
    PyObject* kernel_args;

    if (!PyArg_ParseTuple(args, "OiiiiiiiO", &pso_capsule,
                          &grid_x, &grid_y, &grid_z,
                          &threads_x, &threads_y, &threads_z,
                          &shared_memory, &kernel_args)) {
        return NULL;
    }

    if (metal_init() != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Metal not initialized");
        return NULL;
    }

    id<MTLComputePipelineState> pso =
        (__bridge id<MTLComputePipelineState>)PyCapsule_GetPointer(pso_capsule, "metal_pso");
    if (pso == nil) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid pipeline state object");
        return NULL;
    }

    id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pso];

    // Set threadgroup memory if needed
    if (shared_memory > 0) {
        [encoder setThreadgroupMemoryLength:shared_memory atIndex:0];
    }

    // Set kernel arguments (buffers)
    if (PyTuple_Check(kernel_args)) {
        Py_ssize_t nargs = PyTuple_Size(kernel_args);
        for (Py_ssize_t i = 0; i < nargs; i++) {
            PyObject* arg = PyTuple_GetItem(kernel_args, i);
            if (PyLong_Check(arg)) {
                // Integer argument - set as bytes
                int32_t val = (int32_t)PyLong_AsLong(arg);
                [encoder setBytes:&val length:sizeof(val) atIndex:i];
            }
            // Buffer arguments would be handled via MTLBuffer capsules
        }
    }

    MTLSize gridSize = MTLSizeMake(grid_x, grid_y, grid_z);
    MTLSize tgSize = MTLSizeMake(threads_x, threads_y, threads_z);

    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];

    Py_RETURN_NONE;
}

static PyMethodDef MetalMethods[] = {
    {"init", py_metal_init, METH_NOARGS, "Initialize Metal device"},
    {"get_device_name", py_get_device_name, METH_NOARGS, "Get Metal device name"},
    {"get_gpu_family", py_get_gpu_family, METH_NOARGS, "Get Apple GPU family"},
    {"get_max_threadgroup_memory", py_get_max_threadgroup_memory, METH_NOARGS,
     "Get max threadgroup memory size"},
    {"get_max_threads_per_threadgroup", py_get_max_threads_per_threadgroup, METH_NOARGS,
     "Get max threads per threadgroup"},
    {"load_binary", py_load_binary, METH_VARARGS, "Load a metallib/MSL binary"},
    {"dispatch", py_dispatch, METH_VARARGS, "Dispatch a Metal compute kernel"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef metalmodule = {
    PyModuleDef_HEAD_INIT,
    "metal_utils",
    "Metal runtime utilities for Triton",
    -1,
    MetalMethods
};

PyMODINIT_FUNC PyInit_metal_utils(void) {
    return PyModule_Create(&metalmodule);
}
