#include <map>
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

extern std::map<size_t, int64_t> i64scalar_map;

class RegisterScalarOp : public OpKernel {
public:
  explicit RegisterScalarOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("id", &id_));
  }

  void Compute(OpKernelContext* context) override {
    // fetch input
    const Tensor& x = context->input(0);
    const int32* x_data = (const int32*)x.tensor_data().data();
    const int32 x_rank = x.dims();
    OP_REQUIRES(context, x_rank == 0, errors::InvalidArgument("Input must be a scalar"));
    i64scalar_map[id_] = *x_data;
    context->set_output(0, x);
  }

private:
  int id_;
};


REGISTER_KERNEL_BUILDER(Name("RegisterScalar")
                        .HostMemory("x")
                        .Device(DEVICE_CPU), RegisterScalarOp);
REGISTER_OP("RegisterScalar")
  .Input("x: int32")
  .Output("y: int32")
  .Attr("id: int")
;
