#include <iostream>

#include "triton/driver/buffer.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/runtime/jit.h"
#include "triton/dnn/blocksparse/dot.h"

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/allocation_description.pb.h"

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using GPUDevice = Eigen::GpuDevice;


Status XpropShape(InferenceContext* ctx)
{
  int    K; TF_RETURN_IF_ERROR(ctx->GetAttr(   "K",    &K));
  int axis; TF_RETURN_IF_ERROR(ctx->GetAttr("axis", &axis));

  // C ==> K
  ShapeHandle x = ctx->input(0);
  int rank = ctx->Rank(x);
  //printf("XpropShape: %d\n", rank);
  if (rank > 0)
  {
    std::vector<DimensionHandle> shape;
    shape.reserve(rank);
    for (int i = 0; i < rank; i++)
      shape.push_back(i == axis ? ctx->MakeDim(K) : ctx->Dim(x, i));
    ctx->set_output(0, ctx->MakeShape(shape));
  }
  else
    ctx->set_output(0, ctx->UnknownShape());
  ctx->set_output(1, ctx->UnknownShape());
  return Status::OK();
}


typedef struct bsmm_params
{
  const int* Lut;
  const float* Gate;
  int* Lock;
  int blocks;
  int bsize;
  int segments;
  int locks;
  int C;
  int K;
  int N;
  int shared;
  int pcount;
  uint blk_a;
  uint blk_A;
  uint blk_b;
  uint blk_B;
  float alpha;
  float beta;
  CUstream stream;
} bsmm_params;

template<triton::dnn::blocksparse::op_t OP, typename T>
class BlocksparseMatmulOp : public OpKernel {
public:
  explicit BlocksparseMatmulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("segments", &params_.segments));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("locks",    &params_.locks   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blocks",   &params_.blocks  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bsize",    &params_.bsize  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("C",        &params_.C       ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("K",        &params_.K       ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared",   &params_.shared  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha",    &params_.alpha   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beta",     &params_.beta    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("gated_dw", &gated_dw_       ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis",     &axis_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bench",    &bench_));
    OP_REQUIRES(ctx, params_.K < params_.bsize*65536, errors::InvalidArgument("K < bsize*65536"));
    OP_REQUIRES(ctx, params_.C < params_.bsize*65536, errors::InvalidArgument("C < bsize*65536"));
    params_.pcount = 1;
    params_.blk_A  = 0;
    is_gpu_ = ctx->device_type() == DEVICE_GPU;
    if (bench_) {
      repeat_ = bench_;
      flops_  = (float)(params_.blocks * params_.bsize*params_.bsize);
      const char* op = "FPROP";
      sprintf(bench_string_, "%s %02d-%d C:%05d K:%05d blks:%d", op, params_.bsize, axis_, params_.C, params_.K, params_.blocks);
    }
  }

  void Compute(OpKernelContext* context){
    // get device/stream
    GPUDevice device =  context->eigen_device<GPUDevice>();
    triton::driver::cu_stream sstream(device.stream(), false);
    triton::driver::context* ctx = sstream.context();
    triton::driver::stream* stream = &sstream;
    // get inputs
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    const Tensor& lut = context->input(2);
    // allocate c
    TensorShape shape_c;
    int N     = 1;
    int rank_a = a.dims();
    for (int i = 0; i < rank_a; i++)
      if (i != axis_) {
        shape_c.AddDim(a.dim_size(i));
        N *= a.dim_size(i);
      }
      else
        shape_c.AddDim(params_.K);
    Tensor* c = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape_c, &c));
    // wrap tensorflow handles
    triton::driver::cu_buffer da(ctx, a.tensor_data().size(), (CUdeviceptr)a.tensor_data().data(), false);
    triton::driver::cu_buffer db(ctx, b.tensor_data().size(), (CUdeviceptr)b.tensor_data().data(), false);
    triton::driver::cu_buffer dc(ctx, c->tensor_data().size(), (CUdeviceptr)c->tensor_data().data(), false);
    triton::driver::cu_buffer dlut(ctx, lut.tensor_data().size(), (CUdeviceptr)lut.tensor_data().data(), false);
    // create profile
    triton::dnn::blocksparse::dot dot(N, params_.K, params_.segments, params_.C, "fp16", params_.bsize, params_.locks, params_.blocks, OP);
    // blocksparse matmul
    triton::dnn::base* op = dot.enqueue(stream, {&da, &db, &dc, &dlut}, triton::dnn::FULL_TUNING);
    triton::driver::buffer* locks_buffer = ((triton::dnn::blocksparse::dot*)op)->get_locks();
    Tensor *tmp = nullptr;
    TensorShape tmp_shapes;
    tmp_shapes.AddDim(locks_buffer->size() / 4);
    OP_REQUIRES_OK(context, context->allocate_output(1, tmp_shapes, &tmp));
  }

private:
  bsmm_params params_;
  int   axis_, bench_, repeat_, SMs_, major_, grid_n_;
  float flops_;
  bool  gated_dw_, is_gpu_;
  char bench_string_[256];
};

REGISTER_OP("TritonBlocksparseMatmul")
.Input("x: T")
.Input("w: T")
.Input("lut: int64")
.Input("lut_dx: int64")
.Input("lut_dw: int64")
.Input("gate: ngate * float")
.Output("y: T")
.Output("temp: int32")
.Attr("T: {half, float, bfloat16}")
.Attr("blocks: int >=0")
.Attr("bsize: int")
.Attr("segments: int = 0")
.Attr("segments_dx: int = 0")
.Attr("locks: int = 0")
.Attr("locks_dx: int = 0")
.Attr("axis: int = 1")
.Attr("C: int >=0")
.Attr("K: int >=0")
.Attr("shared: int = 0")
.Attr("shared_dx: int = 0")
.Attr("alpha: float = 1.0")
.Attr("beta: float = 0.0")
.Attr("gated_dw: bool = false")
.Attr("gate_grad: bool = false")
.Attr("bench: int = 0")
.Attr("ngate: int >= 0")
.SetShapeFn(XpropShape)
.Doc(R"doc(
     Multiply the matrix "a" by the blocksparse matrix "b".
     )doc");

REGISTER_KERNEL_BUILDER(Name("TritonBlocksparseMatmul").Device(DEVICE_GPU).TypeConstraint<float>("T"), BlocksparseMatmulOp<triton::dnn::blocksparse::FPROP, float>);
REGISTER_KERNEL_BUILDER(Name("TritonBlocksparseMatmul").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), BlocksparseMatmulOp<triton::dnn::blocksparse::FPROP, Eigen::half>);


REGISTER_OP("TritonBlocksparseMatmulDX")
    .Input("dy: T")
    .Input("w: T")
    .Input("lut: int64")
    .Input("gate: ngate * float")
    .Output("dx: T")
    .Output("temp: int32")
    .Attr("T: {half, float, bfloat16}")
    .Attr("blocks: int >=0")
    .Attr("bsize: int")
    .Attr("segments: int = 0")
    .Attr("locks: int = 0")
    .Attr("axis: int = 1")
    .Attr("C: int >=0")
    .Attr("K: int >=0")
    .Attr("shared: int = 0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.0")
    .Attr("gated_dw: bool = false")
    .Attr("gate_grad: bool = false")
    .Attr("bench: int = 0")
    .Attr("ngate: int >= 0")
    .SetShapeFn(XpropShape)
    .Doc(R"doc(
Multiply the matrix "a" by the blocksparse matrix "b".
)doc");

REGISTER_KERNEL_BUILDER(Name("TritonBlocksparseMatmulDX").Device(DEVICE_GPU).TypeConstraint<float>("T"),BlocksparseMatmulOp<triton::dnn::blocksparse::BPROP, float>);
REGISTER_KERNEL_BUILDER(Name("TritonBlocksparseMatmulDX").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),BlocksparseMatmulOp<triton::dnn::blocksparse::BPROP, Eigen::half>);

