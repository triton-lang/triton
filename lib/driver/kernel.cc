/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <string.h>
#include "triton/driver/kernel.h"
#include "triton/driver/buffer.h"

namespace triton
{

namespace driver
{


/* ------------------------ */
//         Base             //
/* ------------------------ */

kernel::kernel(driver::module *program, CUfunction fn, bool has_ownership):
  polymorphic_resource(fn, has_ownership), program_(program){
}


kernel::kernel(driver::module *program, host_function_t fn, bool has_ownership):
  polymorphic_resource(fn, has_ownership), program_(program){
}

kernel* kernel::create(driver::module* program, const char* name) {
    switch(program->backend()){
    case CUDA: return new cu_kernel(program, name);
    case Host: return new host_kernel(program, name);
    default: throw std::runtime_error("unknown backend");
    }
}

driver::module* kernel::module() {
  return program_;
}

/* ------------------------ */
//         Host             //
/* ------------------------ */

host_kernel::host_kernel(driver::module* program, const char *name): kernel(program, host_function_t(), true) {
  hst_->fn = program->hst()->functions.at(name);
}

/* ------------------------ */
//         CUDA             //
/* ------------------------ */

cu_kernel::cu_kernel(driver::module *program, const char * name) : kernel(program, CUfunction(), true) {
  dispatch::cuModuleGetFunction(&*cu_, *program->cu(), name);
  dispatch::cuFuncSetCacheConfig(*cu_, CU_FUNC_CACHE_PREFER_SHARED);
  // properties
  int shared_total, shared_optin, shared_static;
  int n_spills, n_reg;
  CUdevice dev;
  dispatch::cuCtxGetDevice(&dev);
  dispatch::cuDeviceGetAttribute(&shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, dev);
  dispatch::cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev);
  dispatch::cuFuncGetAttribute(&shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, *cu_);
  dispatch::cuFuncGetAttribute(&n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,  *cu_);
  dispatch::cuFuncGetAttribute(&n_reg, CU_FUNC_ATTRIBUTE_NUM_REGS, *cu_);
  if (shared_optin > 49152){
//      std::cout << "dynamic shared memory " << shared_optin << " " << shared_static << std::endl;
      dispatch::cuFuncSetAttribute(*cu_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin - shared_static);
  }
}

}

}

