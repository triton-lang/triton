#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class AllocEmptyOp : public OpKernel {
 public:
  explicit AllocEmptyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // fetch input
    const Tensor& x = context->input(0);
    const int32* x_data = (const int32*)x.tensor_data().data();
    // allocate output
    Tensor* y = NULL;
    int32 x_rank = x.dims();
    OP_REQUIRES(context, x_rank == 1, errors::InvalidArgument("Input tensor must be 1D"));
    int32 y_rank = x.dim_size(0);
    TensorShape y_shapes;
    for(size_t i = 0; i < y_rank; i++)
      y_shapes.AddDim(x_data[i]);
    OP_REQUIRES_OK(context, context->allocate_output(0, y_shapes, &y));
  }
};


REGISTER_KERNEL_BUILDER(Name("AllocEmpty").HostMemory("x").Device(DEVICE_CPU).Device(DEVICE_GPU), AllocEmptyOp);
REGISTER_OP("AllocEmpty")
  .Input("x: int32")
  .Attr("T : {bool, int8, int16, int32, int64, float16, float32, float64}")
  .Output("y: T")
;
