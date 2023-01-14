#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "add_kernel0.h"

void errMsg(char* msg, int code) {
  if (code != CUDA_SUCCESS) {
   printf("%s with %d\n", msg, code); 
  }
}

#define CHECK_CUDA(code, msg) errMsg(msg, code)

#define GET_DEVICE(id) 

void arange(float *arr, size_t start, size_t size) {

  float val = (float)start;

  for (size_t i = 0; i < size; i++) {
   arr[i] = val;
   val++;
    
  }
}

#define AllocFloatVec(N) (float *)malloc(N * sizeof(float));
#define cuAllocFloatVec(N,ptr) CHECK_CUDA(cuMemAlloc(ptr, N * sizeof(float)), "Allocate Float");

#define VectoDevice(N, src, dst) CHECK_CUDA(cuMemcpyHtoD(dst, src, N*sizeof(float)), "Copy HtoD"); 
#define DevicetoVec(N, src, dst) CHECK_CUDA(cuMemcpyDtoH(dst, src, N*sizeof(float)), "Copy DtoH"); 
#define DEBUG(ptr, msg) printf("%s %p\n", msg, (void *)ptr);

#define VEC_SIZE 160000000

void main() {

  int device_id = 0;
  int max_shared_mem;

  float *u =  AllocFloatVec(VEC_SIZE);
  float *v =  AllocFloatVec(VEC_SIZE);
  float *out =  AllocFloatVec(VEC_SIZE);
  arange(u, 0, VEC_SIZE);
  arange(v, 0, VEC_SIZE);

  CHECK_CUDA(cuInit(0), "Cuda init failed");

  CUdevice device;
  CHECK_CUDA(cuDeviceGet(&device, device_id), "Get device failed");
  cuDeviceGetAttribute(&max_shared_mem,
                       CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                       device);
  
  printf("Max shared mem: %u \n", max_shared_mem);
  CUcontext ctx;
  cuCtxCreate(&ctx, CU_CTX_SCHED_AUTO, device);
  
  CUdeviceptr u_cu, v_cu, out_cu;
  cuAllocFloatVec(VEC_SIZE, &u_cu);
  cuAllocFloatVec(VEC_SIZE, &v_cu);
  cuAllocFloatVec(VEC_SIZE, &out_cu);

  VectoDevice(VEC_SIZE, u, u_cu);
  VectoDevice(VEC_SIZE, v, v_cu);


  GridWarps g = {32, 1, 1, 3};
  CUstream stream;
  CHECK_CUDA(cuStreamCreate(&stream, CU_STREAM_DEFAULT),"stream creation");

  uint32_t n_elem = VEC_SIZE;
  CHECK_CUDA(add_kernel0(stream, g, u_cu, v_cu, out_cu, n_elem), "kernel run");
  DevicetoVec(VEC_SIZE, out_cu, out);
  printf("Out value is %f Expected 30\n", out[15]);
  printf("Out value is %f Expected 2022\n", out[1011]);

}
