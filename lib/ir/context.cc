#include "triton/ir/context_impl.h"
#include "triton/ir/context.h"
#include "triton/ir/type.h"

namespace triton{
namespace ir{

//===----------------------------------------------------------------------===//
//                               context implementation
//===----------------------------------------------------------------------===//

context_impl::context_impl(context &ctx)
    : void_ty(ctx, type::VoidTyID),
      label_ty(ctx, type::LabelTyID),
      // floating point
      fp8_ty(ctx, type::FP8TyID),
      fp16_ty(ctx, type::FP16TyID),
      bf16_ty(ctx, type::BF16TyID),
      fp32_ty(ctx, type::FP32TyID),
      fp64_ty(ctx, type::FP64TyID),
      // integers
      int1_ty(ctx, 1),
      int8_ty(ctx, 8),
      int16_ty(ctx, 16),
      int32_ty(ctx, 32),
      int64_ty(ctx, 64),
      int128_ty(ctx, 128) {}

//===----------------------------------------------------------------------===//
//                                    context
//===----------------------------------------------------------------------===//

context::context():
  p_impl(std::make_shared<context_impl>(*this)) {

}


}
}
