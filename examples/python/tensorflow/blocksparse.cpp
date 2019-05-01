#include <iostream>

#include "triton/driver/buffer.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/jit.h"

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using GPUDevice = Eigen::GpuDevice;


const char* src =
R"(
const tunable int32 TM = {16, 32, 64, 128};
const tunable int32 TN = {16, 32, 64, 128};
const tunable int32 TK = {8};
const tunable int32 GZ = {1};

void bsmm (restrict read_only fp32 *A, restrict read_only fp32 *B, fp32 *C,
           int32 M, int32 N, int32 K,
           int32 lda, int32 ldb, int32 ldc,
           int32 *locks, int32 grid0, int32 grid1) {

}
)";

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


REGISTER_OP("BlocksparseMatmul")
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


typedef struct bsmm_params
{
    const int* Lut;
    const float* Gate;
    int* Lock;
    //float4* Scratch;
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
  }

private:
  bsmm_params params_;
  int   axis_, bench_, repeat_, SMs_, major_, grid_n_;
  float flops_;
  bool  gated_dw_, is_gpu_;
  char bench_string_[256];
};

REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmul").Device(DEVICE_GPU).TypeConstraint<float>("T"), BlocksparseMatmulOp);
