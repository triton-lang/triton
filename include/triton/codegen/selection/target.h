#ifndef TDL_INCLUDE_IR_CODEGEN_TARGET_H
#define TDL_INCLUDE_IR_CODEGEN_TARGET_H

#include <map>
#include <set>
#include <vector>
#include "llvm/IR/IRBuilder.h"

namespace llvm{
class Instruction;
class Value;
class Module;
class LLVMContext;
class Function;
}

namespace triton{
namespace codegen{

class target {
public:
  target(bool is_gpu): is_gpu_(is_gpu){}
  virtual ~target() {}
  virtual void set_kernel(llvm::IRBuilder<>& builder, llvm::LLVMContext &ctx, llvm::Module *module, llvm::Function* fn) = 0;
  virtual llvm::Instruction* add_barrier(llvm::Module *module, llvm::IRBuilder<>& builder) = 0;
  virtual llvm::Instruction* add_memfence(llvm::Module *module, llvm::IRBuilder<>& builder) = 0;
  virtual llvm::Value* get_global_offset(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned stride, unsigned ax) = 0;
  virtual llvm::Value* get_local_id(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned ax) = 0;
  virtual llvm::Value* get_block_id(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned ax) = 0;
  virtual llvm::Value* get_num_blocks(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned ax) = 0;
  bool is_gpu() const;

private:
  bool is_gpu_;
};

class amd_cl_target: public target {
public:
  amd_cl_target(): target(true){}
  void set_kernel(llvm::IRBuilder<>& builder, llvm::LLVMContext &ctx, llvm::Module *module, llvm::Function* fn);
  llvm::Instruction* add_barrier(llvm::Module *module, llvm::IRBuilder<>& builder);
  llvm::Instruction* add_memfence(llvm::Module *module, llvm::IRBuilder<>& builder);
  llvm::Value* get_global_offset(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned stride, unsigned ax);
  llvm::Value* get_local_id(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned ax);
  llvm::Value* get_block_id(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned ax);
  llvm::Value* get_num_blocks(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned ax);
};

class nvidia_cu_target: public target {
public:
  nvidia_cu_target(): target(true){}
  void set_kernel(llvm::IRBuilder<>& builder, llvm::LLVMContext &ctx, llvm::Module *module, llvm::Function* fn);
  llvm::Instruction* add_barrier(llvm::Module *module, llvm::IRBuilder<>& builder);
  llvm::Instruction* add_memfence(llvm::Module *module, llvm::IRBuilder<>& builder);
  llvm::Value* get_global_offset(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned stride, unsigned ax);
  llvm::Value* get_local_id(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned ax);
  llvm::Value* get_block_id(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned ax);
  llvm::Value* get_num_blocks(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned ax);
};

class cpu_target: public target {
public:
  cpu_target(): target(false){}
  void set_kernel(llvm::IRBuilder<>& builder, llvm::LLVMContext &ctx, llvm::Module *module, llvm::Function* fn);
  llvm::Instruction* add_barrier(llvm::Module *module, llvm::IRBuilder<>& builder);
  llvm::Instruction* add_memfence(llvm::Module *module, llvm::IRBuilder<>& builder);
  llvm::Value* get_global_offset(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned stride, unsigned ax);
  llvm::Value* get_local_id(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned ax);
  llvm::Value* get_block_id(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned ax);
  llvm::Value* get_num_blocks(llvm::Module *module, llvm::IRBuilder<>& builder, unsigned ax);
};

}
}

#endif
