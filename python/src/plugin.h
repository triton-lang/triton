#ifndef FLAGTREE_PLUGIN_H
#define FLAGTREE_PLUGIN_H

#include <cassert>
#include <dlfcn.h>
#include <iostream>
#include <string>

#define DEFINE_LOAD_FUNC(symbol_name)                                          \
  static symbol_name##Func load_##symbol_name##_func(const char *backend_name, \
                                                     const char *func_name) {  \
    void *symbol = load_backend_symbol(backend_name, func_name);               \
    return reinterpret_cast<symbol_name##Func>(symbol);                        \
  }

#define DEFINE_CALL_LOAD_FUNC(backend_name, symbol_name)                       \
  static auto func = load_##symbol_name##_func(#backend_name, #symbol_name);

#ifdef _WIN32
#define PLUGIN_EXPORT __declspec(dllexport)
#else
#define PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

static void *load_backend_plugin(const char *backend_name) {
  const std::string lib_name = std::string(backend_name) + "TritonPlugin.so";
  void *handle = dlopen(lib_name.c_str(), RTLD_LAZY);
  if (!handle) {
    std::cerr << "Failed to load plugin: " << std::string(dlerror());
    assert(handle);
  }
  return handle;
}

static void *load_backend_symbol(const char *backend_name,
                                 const char *func_name) {
  void *handle = load_backend_plugin(backend_name);
  void *symbol = dlsym(handle, func_name);
  if (!symbol) {
    std::cerr << "Failed to load symbol: " << std::string(dlerror());
    assert(symbol);
  }
  return symbol;
}

static int load_backend_const_int(const char *backend_name,
                                  const char *const_name) {
  void *handle = load_backend_plugin(backend_name);
  void *symbol = dlsym(handle, const_name);
  if (!symbol) {
    std::cerr << "Failed to load symbol: " << std::string(dlerror());
    assert(symbol);
  }
  return *(const int *)symbol;
}

#endif
