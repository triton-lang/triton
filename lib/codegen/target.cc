#include "triton/codegen/target.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/IRBuilder.h"
#include <iostream>

using namespace llvm;

namespace triton{
namespace codegen{

// base


nvidia_cu_target* target::as_nvidia() { 
  return dynamic_cast<nvidia_cu_target*>(this); 
}

bool target::is_gpu() const {
  return is_gpu_;
}

// AMD
void amd_cl_target::set_kernel(IRBuilder<>& builder, LLVMContext &ctx, Module *module, Function* fn) {
  fn->setCallingConv(CallingConv::AMDGPU_KERNEL);
}

Instruction* amd_cl_target::add_barrier(Module *module, IRBuilder<>& builder) {
  Function *barrier = Intrinsic::getDeclaration(module, Intrinsic::amdgcn_s_barrier);
  return builder.CreateIntrinsic(Intrinsic::amdgcn_s_barrier, {}, {});
}

Value* amd_cl_target::get_global_offset(Module *module, IRBuilder<>& builder, unsigned stride, unsigned ax) {
  Value* group_id = get_block_id(module, builder, ax);
  Value* result = builder.CreateMul(builder.getInt32(stride), group_id);
  return result;
}

Instruction* amd_cl_target::add_memfence(Module *module, IRBuilder<>& builder) {
  throw std::runtime_error("not implemented");
}


Value* amd_cl_target::get_block_id(Module *module, IRBuilder<>& builder, unsigned ax) {
  static std::array<Intrinsic::ID, 3> ids = {
    Intrinsic::amdgcn_workgroup_id_x,
    Intrinsic::amdgcn_workgroup_id_y,
    Intrinsic::amdgcn_workgroup_id_z
  };
  Value* group_id = builder.CreateIntrinsic(ids[ax], {}, {});
  return group_id;
}

Value* amd_cl_target::get_num_blocks(Module *module, IRBuilder<>& builder, unsigned ax) {
  throw std::runtime_error("not implemented on AMD");
}

Value* amd_cl_target::get_local_id(Module *module, IRBuilder<>& builder, unsigned ax) {
  static std::array<Intrinsic::ID, 3> ids = {
    Intrinsic::amdgcn_workitem_id_x,
    Intrinsic::amdgcn_workitem_id_y,
    Intrinsic::amdgcn_workitem_id_z
  };
  Function *get_local_id = Intrinsic::getDeclaration(module, ids[ax]);
  return builder.CreateCall(get_local_id, {});
}

// NVIDIA

void nvidia_cu_target::set_kernel(IRBuilder<>& builder, LLVMContext &ctx, Module *module, Function* fn){
  // set metadata
  Metadata *md_args[] = {
    ValueAsMetadata::get(fn),
    MDString::get(ctx, "kernel"),
    ValueAsMetadata::get(builder.getInt32(1))
  };
  module->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(MDNode::get(ctx, md_args));
}

Instruction* nvidia_cu_target::add_barrier(Module *module, IRBuilder<>& builder) {
  Function *barrier = Intrinsic::getDeclaration(module, Intrinsic::nvvm_barrier0);
  return builder.CreateCall(barrier, {});
}

Instruction* nvidia_cu_target::add_memfence(Module *module, IRBuilder<>& builder) {
  Function *barrier = Intrinsic::getDeclaration(module, Intrinsic::nvvm_membar_gl);
  return builder.CreateCall(barrier, {});
}


Value* nvidia_cu_target::get_global_offset(Module *module, IRBuilder<>& builder, unsigned stride, unsigned ax) {
  Value* group_id = get_block_id(module, builder, ax);
  Value* result = builder.CreateMul(builder.getInt32(stride), group_id);
  return result;
}

Value* nvidia_cu_target::get_block_id(Module *module, IRBuilder<>& builder, unsigned ax) {
  static std::array<Intrinsic::ID, 3> cta_ids = {
    Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
    Intrinsic::nvvm_read_ptx_sreg_ctaid_y,
    Intrinsic::nvvm_read_ptx_sreg_ctaid_z
  };
  Value* cta_id = builder.CreateIntrinsic(cta_ids[ax], {}, {});
  return cta_id;
}

Value* nvidia_cu_target::get_local_id(Module *module, IRBuilder<>& builder, unsigned ax) {
  static std::array<Intrinsic::ID, 3> ids = {
    Intrinsic::nvvm_read_ptx_sreg_tid_x,
    Intrinsic::nvvm_read_ptx_sreg_tid_y,
    Intrinsic::nvvm_read_ptx_sreg_tid_z
  };
  Function *get_local_id = Intrinsic::getDeclaration(module, ids[ax]);
  return builder.CreateCall(get_local_id, {});
}

Value* nvidia_cu_target::get_num_blocks(Module *module, IRBuilder<>& builder, unsigned ax) {
  static std::array<Intrinsic::ID, 3> ids = {
    Intrinsic::nvvm_read_ptx_sreg_nctaid_x,
    Intrinsic::nvvm_read_ptx_sreg_nctaid_y,
    Intrinsic::nvvm_read_ptx_sreg_nctaid_z
  };
  return builder.CreateIntrinsic(ids[ax], {}, {});
}

// CPU

void cpu_target::set_kernel(IRBuilder<>& builder, LLVMContext &ctx, Module *module, Function* fn) {
  // normal cpu functions can be kernels
}

Instruction* cpu_target::add_barrier(Module *module, IRBuilder<>& builder) {
  // no barrier on CPU
  return (Instruction*)builder.CreateAdd(builder.getInt32(0), builder.getInt32(0));
}

Instruction* cpu_target::add_memfence(Module *module, IRBuilder<>& builder) {
  // no barrier on CPU
  return (Instruction*)builder.CreateAdd(builder.getInt32(0), builder.getInt32(0));
}


Value* cpu_target::get_block_id(Module *module, llvm::IRBuilder<> &builder, unsigned ax) {
  const Function *fn = builder.GetInsertBlock()->getParent();
  size_t num_params = fn->getFunctionType()->getNumParams();
  static std::array<const Argument*, 3> ids = {
    fn->arg_begin() + num_params - 3,
    fn->arg_begin() + num_params - 2,
    fn->arg_begin() + num_params - 1
  };
  return (Argument*)ids[ax];
}

Value* cpu_target::get_num_blocks(Module *module, IRBuilder<>& builder, unsigned ax) {
  throw std::runtime_error("not implemented");
}


Value* cpu_target::get_global_offset(Module *module, IRBuilder<>& builder, unsigned stride, unsigned ax) {
  Value* result = builder.CreateMul(builder.getInt32(stride), get_block_id(module, builder, ax));
  return result;
}

Value* cpu_target::get_local_id(Module *module, IRBuilder<>& builder, unsigned ax) {
  return builder.getInt32(0);
}

}
}
