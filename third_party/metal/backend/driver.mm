#include "torch/csrc/autograd/python_variable.h"
#include "torch/mps.h"
#include <ATen/native/mps/OperationUtils.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <memory>
#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace {

struct MetalKernel {
  id<MTLLibrary> lib;
  id<MTLComputePipelineState> pipeline;
  NSUInteger maxThreads;
};

std::mutex cacheMutex;
std::unordered_map<int, std::shared_ptr<MetalKernel>> idToKernelMap;
std::unordered_map<std::string, int> keyToIdMap;
int nextId = 1;

id<MTLLibrary> createLib(id<MTLDevice> dev, const std::string &name,
                         const std::string &metallibBytes) {
  // write metallib bytes to temp file and create MTLLibrary from that file
  NSData *data = [NSData dataWithBytes:metallibBytes.data()
                                length:metallibBytes.size()];
  NSString *tmpDir = NSTemporaryDirectory();
  NSString *filename =
      [[NSUUID UUID].UUIDString stringByAppendingPathExtension:@"metallib"];
  NSString *filepath = [tmpDir stringByAppendingPathComponent:filename];
  NSURL *fileUrl = [NSURL fileURLWithPath:filepath];
  NSError *error = nil;
  if (![data writeToURL:fileUrl options:NSDataWritingAtomic error:&error]) {
    if (error != nil) {
      throw std::runtime_error(
          "failed to write metallib for " + name + ": " +
          std::string([[error localizedDescription] UTF8String]));
    }
    throw std::runtime_error("failed to write metallib for " + name);
  }

  error = nil;
  id<MTLLibrary> lib = [dev newLibraryWithURL:fileUrl error:&error];
  [[NSFileManager defaultManager] removeItemAtURL:fileUrl error:nil];
  if (lib != nil) {
    return lib;
  }
  if (error != nil) {
    throw std::runtime_error(
        "failed to create lib for " + name + ": " +
        std::string([[error localizedDescription] UTF8String]));
  }
  throw std::runtime_error("failed to create lib for " + name);
}

// create cache key for this kernel
std::string makeKernelKey(const std::string &name, const py::bytes &metallib) {
  std::string bytes = metallib;
  std::string key;
  key.reserve(name.size() + 1 + bytes.size());
  key.append(name);
  key.push_back('\0');
  key.append(bytes);
  return key;
}

// create MetalKernel for this kernel to store in the global cache
std::shared_ptr<MetalKernel> createBinary(const std::string &name,
                                          const py::bytes &metallib) {
  std::string metallibBytes = metallib;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (dev == nil) {
    throw std::runtime_error("failed to create metal device");
  }

  NSError *error = nil;
  id<MTLLibrary> lib = createLib(dev, name, metallibBytes);

  NSString *functionName = [NSString stringWithUTF8String:name.c_str()];
  id<MTLFunction> func = [lib newFunctionWithName:functionName];
  if (func == nil) {
    throw std::runtime_error(name + " not found");
  }

  id<MTLComputePipelineState> pipeline =
      [dev newComputePipelineStateWithFunction:func error:&error];
  if (pipeline == nil) {
    throw std::runtime_error([[error localizedDescription] UTF8String]);
  }

  return std::make_shared<MetalKernel>(
      MetalKernel{lib, pipeline, [pipeline maxTotalThreadsPerThreadgroup]});
}

template <typename T> static std::vector<uint8_t> packScalar(py::object arg) {
  T v = arg.cast<T>();
  std::vector<uint8_t> bytes(sizeof(T));
  std::memcpy(bytes.data(), &v, sizeof(T));
  return bytes;
}

static std::vector<uint8_t> scalarToBytes(const std::string &ty,
                                          py::object arg) {
  if (ty == "i1" || ty == "i8")
    return packScalar<int8_t>(arg);
  if (ty == "u8")
    return packScalar<uint8_t>(arg);
  if (ty == "i16")
    return packScalar<int16_t>(arg);
  if (ty == "u16")
    return packScalar<uint16_t>(arg);
  if (ty == "i32")
    return packScalar<int32_t>(arg);
  if (ty == "u32")
    return packScalar<uint32_t>(arg);
  if (ty == "i64")
    return packScalar<int64_t>(arg);
  if (ty == "u64")
    return packScalar<uint64_t>(arg);
  if (ty == "fp32" || ty == "f32")
    return packScalar<float>(arg);
  if (ty == "fp64")
    return packScalar<double>(arg);
  throw std::runtime_error("unsupported scalar type: " + ty);
}

// kernel arg: MTLBuffer or inline bytes
struct KernelArg {
  id<MTLBuffer> buffer = nil; // use setBuffer
  std::vector<uint8_t> bytes; // use setBytes
};

void launchKernel(const std::tuple<int, int, int> &grid, int kernelId,
                  py::dict signature, py::tuple args, py::object kernelMetadata,
                  int numWarps, int warpSize) {
  std::shared_ptr<MetalKernel> kernel = idToKernelMap.at(kernelId);
  id<MTLComputePipelineState> pipelineState = kernel->pipeline;

  // extract args while holding GIL because dispatch_sync runs on different
  // thread
  std::vector<KernelArg> kernelArgs;
  int argIdx = 0;
  for (auto item : signature) {
    std::string ty = item.second.cast<std::string>();
    if (ty == "constexpr") {
      argIdx++;
      continue;
    }
    py::object arg = args[argIdx++];
    KernelArg ka;
    if (ty[0] == '*') {
      at::Tensor t = THPVariable_Unpack(arg.ptr());
      ka.buffer = at::native::mps::getMTLBufferStorage(t);
      kernelArgs.push_back(std::move(ka));

      // inject one implicit stride arg with all dims packed
      KernelArg sa;
      sa.bytes.resize(t.dim() * sizeof(int64_t));
      for (int d = 0; d < t.dim(); d++) {
        int64_t stride = t.stride(d);
        std::memcpy(sa.bytes.data() + d * sizeof(int64_t), &stride,
                    sizeof(int64_t));
      }
      kernelArgs.push_back(std::move(sa));
    } else {
      ka.bytes = scalarToBytes(ty, arg);
      kernelArgs.push_back(std::move(ka));
    }
  }

  auto [gx, gy, gz] = grid;
  MTLSize threadgroups = MTLSizeMake(gx, gy, gz);
  MTLSize threadsPerTG = MTLSizeMake(numWarps * warpSize, 1, 1);

  dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
  dispatch_sync(serialQueue, ^{
    id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pipelineState];

    for (int i = 0; i < (int)kernelArgs.size(); i++) {
      const KernelArg &ka = kernelArgs[i];
      if (ka.buffer != nil) {
        [encoder setBuffer:ka.buffer offset:0 atIndex:i];
      } else {
        [encoder setBytes:ka.bytes.data() length:ka.bytes.size() atIndex:i];
      }
    }

    [encoder dispatchThreadgroups:threadgroups
            threadsPerThreadgroup:threadsPerTG];
    [encoder endEncoding];
  });
}

} // namespace

PYBIND11_MODULE(_metal_driver, m) {
  m.def("get_mtl_buffer", [](py::object obj) -> uintptr_t {
    at::Tensor t = THPVariable_Unpack(obj.ptr());
    id<MTLBuffer> buf = at::native::mps::getMTLBufferStorage(t);
    return reinterpret_cast<uintptr_t>(buf);
  });

  m.def("get_command_buffer", []() -> uintptr_t {
    id<MTLCommandBuffer> buf = torch::mps::get_command_buffer();
    return reinterpret_cast<uintptr_t>(buf);
  });

  m.def("load_binary", [](const std::string &name, py::bytes metallib) {
    std::lock_guard<std::mutex> lock(cacheMutex);
    std::string key = makeKernelKey(name, metallib);
    int kernel_id = 0;
    std::shared_ptr<MetalKernel> binary;

    auto it = keyToIdMap.find(key);
    if (it != keyToIdMap.end()) {
      // already cached
      kernel_id = it->second;
      binary = idToKernelMap.at(kernel_id);
    } else {
      // create and cache kernel representation
      kernel_id = nextId++;
      binary = createBinary(name, metallib);
      keyToIdMap.emplace(std::move(key), kernel_id);
      idToKernelMap.emplace(kernel_id, binary);
    }

    return py::make_tuple(kernel_id, kernel_id, 0, 0,
                          static_cast<int>(binary->maxThreads));
  });

  m.def("unload_module", [](int kernel_id) {
    std::lock_guard<std::mutex> lock(cacheMutex);
    auto id_it = idToKernelMap.find(kernel_id);
    if (id_it == idToKernelMap.end()) {
      return;
    }

    // search for id in cache and remove kernel
    for (auto key_it = keyToIdMap.begin(); key_it != keyToIdMap.end();
         ++key_it) {
      if (key_it->second == kernel_id) {
        keyToIdMap.erase(key_it);
        break;
      }
    }
    idToKernelMap.erase(id_it);
  });

  m.def("launch_kernel", &launchKernel, py::arg("grid"), py::arg("kernel_id"),
        py::arg("signature"), py::arg("args"), py::arg("kernel_metadata"),
        py::arg("num_warps"), py::arg("warp_size"));
}
