#include <iostream>

#include "triton/driver/buffer.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("BlockSparseGemm")
    .Attr("T: {float}")
    .Input("A: float")
    .Input("B: float")
    .Output("C: float");

class BlockSparseGemmOp : public OpKernel {
 public:
  explicit BlockSparseGemmOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context){
    GPUDevice device =  context->eigen_device<GPUDevice>();
    triton::driver::cu_stream stream(device.stream(), false);
  }

private:
};

REGISTER_KERNEL_BUILDER(Name("BlockSparse").Device(DEVICE_GPU), BlockSparseGemmOp);
