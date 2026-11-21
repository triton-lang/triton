#pragma once

#include <Python.h>
#include <cstddef>
#include <cstdint>

extern "C" void *gsanProfileMalloc(ssize_t size, int device, void *stream);
extern "C" void gsanProfileFree(void *ptr, ssize_t size, int device,
                                void *stream);
bool profileGetLayout(uintptr_t address, uintptr_t &base, size_t &size);
bool registerProfileCUDAAllocatorBindings(PyObject *module);
