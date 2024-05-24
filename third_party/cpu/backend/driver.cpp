//===- driver.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"

#include <cstddef>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;

  return Py_BuildValue("{s:i}", "max_shared_mem", 0);
}

bool getBoolEnv(const std::string &env) {
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return (str == "on" || str == "true" || str == "1");
}

llvm::orc::ThreadSafeContext &getThreadSafeContext() {
  static llvm::orc::ThreadSafeContext tsc;
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    auto context = std::make_unique<llvm::LLVMContext>();
    tsc = llvm::orc::ThreadSafeContext(std::move(context));
  });
  return tsc;
}

std::string llvmErrToString(const llvm::Error &err) {
  std::string res;
  llvm::raw_string_ostream os(res);
  os << err;
  return res;
};

struct CompiledKernel {
  std::unique_ptr<llvm::orc::ExecutionSession> execution_session;
  std::unique_ptr<llvm::DataLayout> data_layout;
  std::unique_ptr<llvm::orc::MangleAndInterner> mangle;
  std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer> object_layer;
  std::unique_ptr<llvm::orc::IRCompileLayer> compiler_layer;
  llvm::orc::JITDylib *dylib = nullptr;

  CompiledKernel() = default;
  CompiledKernel(CompiledKernel &&) = default;

  ~CompiledKernel() {
    if (execution_session)
      llvm::cantFail(execution_session->endSession());
  }
};

std::vector<std::unique_ptr<CompiledKernel>> compiled_kernels;

static PyObject *loadBitcode(PyObject *self, PyObject *args) {
  const char *name;
  int shared;
  PyObject *py_bytes;
  int devId;

  if (!PyArg_ParseTuple(args, "sSii", &name, &py_bytes, &shared, &devId)) {
    std::cerr << "loadBitcode arg parse failed" << std::endl;
    return NULL;
  }

  std::string kernel_name = name;
  size_t binary_size = PyBytes_Size(py_bytes);
  const char *binary_ptr = PyBytes_AsString(py_bytes);

  llvm::LLVMContext context;
  auto buf = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(binary_ptr, binary_size));
  auto mod = llvm::parseBitcodeFile(*buf, context);
  if (!mod) {
    std::cerr << "Failed to parse LLVM bitcode module" << std::endl;
    return NULL;
  }

  if (getBoolEnv("MLIR_ENABLE_DUMP")) {
    llvm::errs() << "********** Loaded Module (kernel_name=" << name
                 << ") **********\n"
                 << **mod << "\n";
  }

  auto init_err = llvm::InitializeNativeTarget();
  if (init_err) {
    std::cerr << "Failed to initialize native target." << std::endl;
    return NULL;
  }

  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  auto self_epc =
      llvm::cantFail(llvm::orc::SelfExecutorProcessControl::Create());

  auto detect_host_res = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!detect_host_res) {
    std::cerr << "Failed to initialize JITTargetMachineBuilder: "
              << llvmErrToString(detect_host_res.takeError());
    return NULL;
  }
  llvm::orc::JITTargetMachineBuilder tmb = std::move(*detect_host_res);

  auto data_layout_res = tmb.getDefaultDataLayoutForTarget();
  if (!data_layout_res) {
    std::cerr << "Failed to initialize data layout: "
              << llvmErrToString(data_layout_res.takeError());
    return NULL;
  }

  CompiledKernel kernel;
  kernel.execution_session =
      std::make_unique<llvm::orc::ExecutionSession>(std::move(self_epc));
  kernel.data_layout =
      std::make_unique<llvm::DataLayout>(std::move(*data_layout_res));
  kernel.mangle = std::make_unique<llvm::orc::MangleAndInterner>(
      *kernel.execution_session, *kernel.data_layout);
  kernel.object_layer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
      *kernel.execution_session,
      []() { return std::make_unique<llvm::SectionMemoryManager>(); });
  kernel.compiler_layer = std::make_unique<llvm::orc::IRCompileLayer>(
      *kernel.execution_session, *kernel.object_layer,
      std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(tmb)));

  auto dylib_res = kernel.execution_session->createJITDylib("<main>");
  if (!dylib_res) {
    std::cerr << "Failed to create initialize JITDylib: "
              << llvmErrToString(dylib_res.takeError());
    return NULL;
  }

  kernel.dylib = &(*dylib_res);
  kernel.dylib->addGenerator(llvm::cantFail(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          kernel.data_layout->getGlobalPrefix())));

  // Compile module.
  (**mod).setDataLayout(*kernel.data_layout);
  llvm::orc::ThreadSafeModule tsm(std::move(*mod), getThreadSafeContext());
  auto err = kernel.compiler_layer->add(*kernel.dylib, std::move(tsm));
  if (err) {
    std::cerr << "Cannot add LLVM module: " << llvmErrToString(err);
    return NULL;
  }

  // Find kernel function pointer.
  auto lookup_res =
      kernel.execution_session->lookup({kernel.dylib}, (*kernel.mangle)(name));
  if (!lookup_res) {
    std::cerr << "Failed to find function " << std::string(name)
              << "\nError: " << llvmErrToString(lookup_res.takeError());
    return NULL;
  }
  uint64_t fn_ptr = lookup_res->getAddress().getValue();

  compiled_kernels.push_back(
      std::make_unique<CompiledKernel>(std::move(kernel)));
  auto *kernel_ptr = compiled_kernels.back().get();

  return Py_BuildValue("(KKii)", reinterpret_cast<uint64_t>(kernel_ptr),
                       reinterpret_cast<uint64_t>(fn_ptr), 0, 0);
}

static PyObject *initContext(PyObject *self, PyObject *args) {
  return Py_BuildValue("(K)", (uint64_t)0);
}

static PyObject *initDevices(PyObject *self, PyObject *args) {
  return Py_BuildValue("(i)", 1);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBitcode, METH_VARARGS,
     "Load provided SPV into ZE driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "cpu_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_cpu_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
