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
      int1_ty(ctx, 1, signedness::SIGNED),
      int8_ty(ctx, 8, signedness::SIGNED),
      int16_ty(ctx, 16, signedness::SIGNED),
      int32_ty(ctx, 32, signedness::SIGNED),
      int64_ty(ctx, 64, signedness::SIGNED),
      int128_ty(ctx, 128, signedness::SIGNED),
      uint8_ty(ctx, 8, signedness::UNSIGNED),
      uint16_ty(ctx, 16, signedness::UNSIGNED),
      uint32_ty(ctx, 32, signedness::UNSIGNED),
      uint64_ty(ctx, 64, signedness::UNSIGNED){

}

//===----------------------------------------------------------------------===//
//                                    context
//===----------------------------------------------------------------------===//

context::context():
  p_impl(std::make_shared<context_impl>(*this)) {

}


}
}
