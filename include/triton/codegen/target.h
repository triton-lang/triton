#ifndef TDL_INCLUDE_IR_CODEGEN_TARGET_H
#define TDL_INCLUDE_IR_CODEGEN_TARGET_H

namespace llvm{
  class Type;
  class Value;
  class Instruction;
  class Constant;
  class LLVMContext;
  class Module;
  class ConstantFolder;
  class IRBuilderDefaultInserter;
  template <typename T, typename Inserter>
  class IRBuilder;
  class ArrayType;
  class Function;
}

// typedefs
namespace triton{
namespace codegen{
  typedef llvm::IRBuilder<llvm::ConstantFolder,
                          llvm::IRBuilderDefaultInserter> Builder;
  typedef llvm::LLVMContext LLVMContext;
  typedef llvm::Type Type;
  typedef llvm::Value Value;
  typedef llvm::Module Module;
  typedef llvm::Instruction Instruction;
  typedef llvm::Constant Constant;
  typedef llvm::ArrayType ArrayType;
  typedef llvm::Function Function;
}
}

namespace triton{
namespace codegen{

class nvidia_cu_target;
class amd_cl_target;

class target {
public:
  target(bool is_gpu): is_gpu_(is_gpu){}
  virtual ~target() {}
  virtual void set_kernel(Builder& builder, LLVMContext &ctx, Module *module, Function* fn) = 0;
  virtual Instruction* add_barrier(Module *module, Builder& builder) = 0;
  virtual Instruction* add_memfence(Module *module, Builder& builder) = 0;
  virtual Value* get_global_offset(Module *module, Builder& builder, unsigned stride, unsigned ax) = 0;
  virtual Value* get_local_id(Module *module, Builder& builder, unsigned ax) = 0;
  virtual Value* get_block_id(Module *module, Builder& builder, unsigned ax) = 0;
  virtual Value* get_num_blocks(Module *module, Builder& builder, unsigned ax) = 0;
  virtual unsigned guaranteed_alignment() = 0;
#ifdef USE_ROCM
  amd_cl_target* as_nvidia();
  amd_cl_target* as_amd();
#else
  nvidia_cu_target* as_nvidia();
#endif
  bool is_gpu() const;

private:
  bool is_gpu_;
};

class amd_cl_target: public target {
public:
  amd_cl_target(): target(true){}
  void set_kernel(Builder& builder, LLVMContext &ctx, Module *module, Function* fn);
  Instruction* add_barrier(Module *module, Builder& builder);
  Instruction* add_memfence(Module *module, Builder& builder);
  Value* get_global_offset(Module *module, Builder& builder, unsigned stride, unsigned ax);
  Value* get_local_id(Module *module, Builder& builder, unsigned ax);
  Value* get_block_id(Module *module, Builder& builder, unsigned ax);
  Value* get_num_blocks(Module *module, Builder& builder, unsigned ax);
  unsigned guaranteed_alignment() { return 16; }
  int sm() { return 0; } // treat as if old CUDA device
};

class nvidia_cu_target: public target {
public:
  nvidia_cu_target(int sm): target(true), sm_(sm){}
  void set_kernel(Builder& builder, LLVMContext &ctx, Module *module, Function* fn);
  Instruction* add_barrier(Module *module, Builder& builder);
  Instruction* add_memfence(Module *module, Builder& builder);
  Value* get_global_offset(Module *module, Builder& builder, unsigned stride, unsigned ax);
  Value* get_local_id(Module *module, Builder& builder, unsigned ax);
  Value* get_block_id(Module *module, Builder& builder, unsigned ax);
  Value* get_num_blocks(Module *module, Builder& builder, unsigned ax);
  int sm() { return sm_; }
  unsigned guaranteed_alignment() { return 16; }

private:
  int sm_;
};

class cpu_target: public target {
public:
  cpu_target(): target(false){}
  void set_kernel(Builder& builder, LLVMContext &ctx, Module *module, Function* fn);
  Instruction* add_barrier(Module *module, Builder& builder);
  Instruction* add_memfence(Module *module, Builder& builder);
  Value* get_global_offset(Module *module, Builder& builder, unsigned stride, unsigned ax);
  Value* get_local_id(Module *module, Builder& builder, unsigned ax);
  Value* get_block_id(Module *module, Builder& builder, unsigned ax);
  Value* get_num_blocks(Module *module, Builder& builder, unsigned ax);
  unsigned guaranteed_alignment() { return 1; }
};

}
}

#endif
