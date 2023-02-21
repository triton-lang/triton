#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Target/LLVMIR/LLVMIRTranslation.h"
#include "triton/Target/PTX/PTXTranslation.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/SourceMgr.h"

#include <Python.h>
#include <cctype>
#include <fstream>
#include <optional>
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace py = pybind11;

enum backend_t {
  HOST,
  CUDA,
  ROCM,
};

void init_triton_runtime(py::module &&m) {
  // wrap backend_t
  py::enum_<backend_t>(m, "backend")
      .value("HOST", HOST)
      .value("CUDA", CUDA)
      // .value("ROCM", ROCM)
      .export_values();
}

/*****************************************************************************/
/* Python bindings for triton::ir                                            */
/*****************************************************************************/

void init_triton_ir(py::module &&m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;

  py::enum_<mlir::triton::CacheModifier>(m, "CACHE_MODIFIER")
      .value("NONE", mlir::triton::CacheModifier::NONE)
      .value("CA", mlir::triton::CacheModifier::CA)
      .value("CG", mlir::triton::CacheModifier::CG)
      .export_values();

  py::enum_<mlir::triton::EvictionPolicy>(m, "EVICTION_POLICY")
      .value("NORMAL", mlir::triton::EvictionPolicy::NORMAL)
      .value("EVICT_FIRST", mlir::triton::EvictionPolicy::EVICT_FIRST)
      .value("EVICT_LAST", mlir::triton::EvictionPolicy::EVICT_LAST)
      .export_values();

  py::enum_<mlir::triton::RedOp>(m, "REDUCE_OP")
      .value("ADD", mlir::triton::RedOp::ADD)
      .value("FADD", mlir::triton::RedOp::FADD)
      .value("MIN", mlir::triton::RedOp::MIN)
      .value("MAX", mlir::triton::RedOp::MAX)
      .value("UMIN", mlir::triton::RedOp::UMIN)
      .value("UMAX", mlir::triton::RedOp::UMAX)
      .value("ARGMIN", mlir::triton::RedOp::ARGMIN)
      .value("ARGMAX", mlir::triton::RedOp::ARGMAX)
      .value("ARGUMIN", mlir::triton::RedOp::ARGUMIN)
      .value("ARGUMAX", mlir::triton::RedOp::ARGUMAX)
      .value("FMIN", mlir::triton::RedOp::FMIN)
      .value("FMAX", mlir::triton::RedOp::FMAX)
      .value("ARGFMIN", mlir::triton::RedOp::ARGFMIN)
      .value("ARGFMAX", mlir::triton::RedOp::ARGFMAX)
      .value("XOR", mlir::triton::RedOp::XOR);

  py::enum_<mlir::triton::RMWOp>(m, "ATOMIC_OP")
      .value("ADD", mlir::triton::RMWOp::ADD)
      .value("FADD", mlir::triton::RMWOp::FADD)
      .value("AND", mlir::triton::RMWOp::AND)
      .value("OR", mlir::triton::RMWOp::OR)
      .value("XOR", mlir::triton::RMWOp::XOR)
      .value("XCHG", mlir::triton::RMWOp::XCHG)
      .value("MAX", mlir::triton::RMWOp::MAX)
      .value("MIN", mlir::triton::RMWOp::MIN)
      .value("UMIN", mlir::triton::RMWOp::UMIN)
      .value("UMAX", mlir::triton::RMWOp::UMAX);

  py::class_<mlir::MLIRContext>(m, "context")
      .def(py::init<>())
      .def("load_triton", [](mlir::MLIRContext &self) {
        self.getOrLoadDialect<mlir::triton::TritonDialect>();
        // we load LLVM because the frontend uses LLVM.undef for
        // some placeholders
        self.getOrLoadDialect<mlir::triton::TritonDialect>();
        self.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        self.getOrLoadDialect<mlir::gpu::GPUDialect>();
      });
  // .def(py::init([](){
  //   mlir::MLIRContext context;
  //   context.getOrLoadDialect<mlir::triton.TritonDialect>();
  //   // TODO: should we return a (raw/unique) pointer here?
  //   return context;
  // }));

  // py::class_<ir::value>(m, "value")
  //     .def("multiple_of", [](ir::value *self, int val) {
  //       if (auto *instr = dynamic_cast<ir::instruction*>(self)) {
  //         instr->set_metadata(ir::metadata::multiple_of, val);
  //       } else
  //         throw std::runtime_error("multiple_of");
  //     })
  //     .def("max_contiguous", [](ir::value *self, int val) {
  //       if (auto *instr = dynamic_cast<ir::instruction*>(self)) {
  //         instr->set_metadata(ir::metadata::max_contiguous, val);
  //       } else
  //         throw std::runtime_error("max_contiguous");
  //     })
  //     .def("set_fdiv_ieee_rounding", [](ir::value *self, bool val) {
  //       if (auto *instr = dynamic_cast<ir::binary_operator*>(self))
  //         instr->set_fdiv_ieee_rounding(val);
  //       else
  //         throw std::runtime_error("set_fdiv_ieee_rounding");
  //     })
  //     .def("ops", [](ir::value *self) {
  //       if (auto *instr = dynamic_cast<ir::instruction*>(self)) {
  //         return instr->ops();
  //       }
  //       throw std::runtime_error("cannot use ops()");
  //     })
  //     .def("replace_all_uses_with", &ir::value::replace_all_uses_with)
  //     .def("erase_from_parent", [](ir::value *self) {
  //       if (auto *instr = dynamic_cast<ir::instruction*>(self))
  //         return instr->erase_from_parent();
  //       throw std::runtime_error("cannot use erase_from_parent");
  //     })
  //     .def_property("name", &ir::value::get_name, &ir::value::set_name)
  //     .def_property_readonly("type", &ir::value::get_type);

  // // // Do we need under in TritonIR ?
  // // py::class_<ir::undef_value, ir::constant>(m, "undef")
  // //     .def("get", &ir::undef_value::get, ret::reference);

  py::class_<mlir::Type>(m, "type")
      .def("is_integer", &mlir::Type::isInteger)
      .def("is_fp16", &mlir::Type::isF16)
      .def("__str__", [](mlir::Type &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  py::class_<mlir::FunctionType>(m, "function_type")
      .def("param_types", [](mlir::FunctionType &self) {
        return std::vector<mlir::Type>(self.getInputs().begin(),
                                       self.getInputs().end());
      });

  py::class_<mlir::Value>(m, "value")
      .def("set_attr",
           [](mlir::Value &self, std::string &name,
              mlir::Attribute &attr) -> void {
             if (mlir::Operation *definingOp = self.getDefiningOp())
               definingOp->setAttr(name, attr);
             else {
               /* issue a warning */
             }
           })
      .def("get_context", &mlir::Value::getContext)
      .def("replace_all_uses_with",
           [](mlir::Value &self, mlir::Value &newValue) {
             self.replaceAllUsesWith(newValue);
           })
      .def("get_type", &mlir::Value::getType);

  py::class_<mlir::BlockArgument, mlir::Value>(m, "block_argument");

  py::class_<mlir::Region>(m, "region")
      .def("get_parent_region", &mlir::Region::getParentRegion, ret::reference)
      .def("size", [](mlir::Region &self) { return self.getBlocks().size(); })
      .def("empty", &mlir::Region::empty);

  py::class_<mlir::Block>(m, "block")
      .def("arg",
           [](mlir::Block &self, int index) -> mlir::BlockArgument {
             return self.getArgument(index);
           })
      .def("add_argument",
           [](mlir::Block &self, mlir::Type ty) {
             auto loc = mlir::UnknownLoc::get(ty.getContext());
             self.addArgument(ty, loc);
           })
      .def("get_num_arguments", &mlir::Block::getNumArguments)
      .def("dump", &mlir::Block::dump)
      .def("move_before", &mlir::Block::moveBefore)
      .def("insert_before", &mlir::Block::insertBefore)
      .def("get_parent", &mlir::Block::getParent, ret::reference)
      .def("merge_block_before",
           [](mlir::Block &self, mlir::Block &dst) {
             // ref: RewriterBase::mergeBlocks()
             if (self.getNumArguments() != 0)
               throw std::runtime_error(
                   "This block has arguments, don't merge");
             dst.getOperations().splice(dst.begin(), self.getOperations());
             self.dropAllUses();
             self.erase();
           })
      .def("replace_use_in_block_with",
           [](mlir::Block &self, mlir::Value &v, mlir::Value &newVal) {
             v.replaceUsesWithIf(newVal, [&](mlir::OpOperand &operand) {
               mlir::Operation *user = operand.getOwner();
               mlir::Block *currentBlock = user->getBlock();
               while (currentBlock) {
                 if (currentBlock == &self)
                   return true;
                 // Move up one level
                 currentBlock =
                     currentBlock->getParent()->getParentOp()->getBlock();
               }
               return false;
             });
           })
      .def("__str__",
           [](mlir::Block &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def("has_terminator",
           [](mlir::Block &self) {
             return !self.empty() &&
                    self.back().hasTrait<mlir::OpTrait::IsTerminator>();
           })
      .def("erase", [](mlir::Block &self) { self.erase(); });

  // using eattr = ir::attribute_kind_t;
  // py::enum_<eattr>(m, "attribute_kind")
  //     .value("readonly", eattr::readonly)
  //     .value("writeonly", eattr::writeonly)
  //     .value("noalias", eattr::noalias)
  //     .value("aligned", eattr::aligned)
  //     .value("multiple_of", eattr::multiple_of)
  //     .value("retune", eattr::retune)
  //     .value("not_implemented", eattr::not_implemented);

  py::class_<mlir::Attribute>(m, "attribute");
  py::class_<mlir::IntegerAttr, mlir::Attribute>(m, "integer_attr");
  py::class_<mlir::BoolAttr, mlir::Attribute>(m, "bool_attr");

  // Ops
  py::class_<mlir::OpState>(m, "OpState")
      .def("set_attr",
           [](mlir::OpState &self, std::string &name,
              mlir::Attribute &attr) -> void { self->setAttr(name, attr); })
      .def(
          "get_num_results",
          [](mlir::OpState &self) -> unsigned { return self->getNumResults(); })
      .def("get_result",
           [](mlir::OpState &self, unsigned idx) -> mlir::Value {
             return self->getResult(idx);
           })
      .def(
          "get_region",
          [](mlir::OpState &self, unsigned idx) -> mlir::Region & {
            return self->getRegion(idx);
          },
          ret::reference)
      .def(
          "get_body",
          [](mlir::scf::ForOp &self, unsigned idx) -> mlir::Block * {
            return self.getBody(idx);
          },
          ret::reference)
      .def("dump", [](mlir::OpState &self) { self->dump(); })
      .def("__str__",
           [](mlir::OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             self->print(os);
             return str;
           })
      .def("append_operand",
           [](mlir::OpState &self, mlir::Value &val) {
             self->insertOperands(self->getNumOperands(), val);
           })
      .def("verify", [](mlir::OpState &self) -> bool {
        return mlir::succeeded(mlir::verify(self.getOperation()));
      });
  // scf Ops
  py::class_<mlir::scf::ForOp, mlir::OpState>(m, "ForOp")
      .def("get_induction_var", &mlir::scf::ForOp::getInductionVar);

  py::class_<mlir::scf::IfOp, mlir::OpState>(m, "IfOp")
      .def("get_then_block", &mlir::scf::IfOp::thenBlock, ret::reference)
      .def("get_else_block", &mlir::scf::IfOp::elseBlock, ret::reference)
      .def("get_then_yield", &mlir::scf::IfOp::thenYield)
      .def("get_else_yield", &mlir::scf::IfOp::elseYield);
  py::class_<mlir::scf::YieldOp, mlir::OpState>(m, "YieldOp");
  py::class_<mlir::scf::WhileOp, mlir::OpState>(m, "WhileOp")
      .def("get_before", &mlir::scf::WhileOp::getBefore, ret::reference)
      .def("get_after", &mlir::scf::WhileOp::getAfter, ret::reference);
  py::class_<mlir::scf::ConditionOp, mlir::OpState>(m, "ConditionOp");

  // dynamic_attr is used to transfer ownership of the MLIR context to the
  // module
  py::class_<mlir::ModuleOp, mlir::OpState>(m, "module", py::dynamic_attr())
      .def("dump", &mlir::ModuleOp::dump)
      .def("str",
           [](mlir::ModuleOp &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def("push_back",
           [](mlir::ModuleOp &self, mlir::FuncOp &funcOp) -> void {
             self.push_back(funcOp);
           })
      .def("has_function",
           [](mlir::ModuleOp &self, std::string &funcName) -> bool {
             if (self.lookupSymbol(funcName))
               return true;
             return false;
           })
      .def("get_function",
           [](mlir::ModuleOp &self, std::string &funcName) -> mlir::FuncOp {
             return self.lookupSymbol<mlir::FuncOp>(funcName);
           })
      .def("get_single_function", [](mlir::ModuleOp &self) -> mlir::FuncOp {
        llvm::SmallVector<mlir::FuncOp> funcs;
        self.walk([&](mlir::FuncOp func) { funcs.push_back(func); });
        if (funcs.size() != 1)
          throw std::runtime_error("Expected a single function");
        return funcs[0];
      });

  m.def("make_attr",
        [](const std::vector<int> &values, mlir::MLIRContext &context) {
          return mlir::DenseIntElementsAttr::get(
                     mlir::RankedTensorType::get(
                         {static_cast<int64_t>(values.size())},
                         mlir::IntegerType::get(&context, 32)),
                     values)
              .cast<mlir::Attribute>();
        });

  m.def(
      "parse_mlir_module",
      [](const std::string &inputFilename, mlir::MLIRContext &context) {
        // initialize registry
        // note: we initialize llvm for undef
        mlir::DialectRegistry registry;
        registry.insert<mlir::triton::TritonDialect,
                        mlir::triton::gpu::TritonGPUDialect,
                        mlir::math::MathDialect, mlir::arith::ArithmeticDialect,
                        mlir::StandardOpsDialect, mlir::scf::SCFDialect>();
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();

        // parse module
        mlir::OwningOpRef<mlir::ModuleOp> module(
            mlir::parseSourceFile(inputFilename, &context));
        // locations are incompatible with ptx < 7.5 !
        module->walk([](mlir::Operation *op) {
          op->setLoc(mlir::UnknownLoc::get(op->getContext()));
        });
        if (!module)
          throw std::runtime_error("Parse MLIR file failed.");

        return module->clone();
      },
      ret::take_ownership);

  py::class_<mlir::FuncOp, mlir::OpState>(m, "function")
      // .def_property_readonly("attrs", &ir::function::attrs)
      // .def("add_attr", &ir::function::add_attr);
      .def("args",
           [](mlir::FuncOp &self, unsigned idx) -> mlir::BlockArgument {
             return self.getArgument(idx);
           })
      .def(
          "add_entry_block",
          [](mlir::FuncOp &self) -> mlir::Block * {
            return self.addEntryBlock();
          },
          ret::reference)
      .def(
          "set_arg_attr",
          [](mlir::FuncOp &self, int arg_no, const std::string &name, int val) {
            // set arg attributes "name" to value "val"
            auto attrTy = mlir::IntegerType::get(self.getContext(), 32);
            self.setArgAttr(arg_no, name, mlir::IntegerAttr::get(attrTy, val));
          },
          ret::reference)
      .def_property_readonly("type", &mlir::FuncOp::getType)
      .def("reset_type", &mlir::FuncOp::setType);

  py::class_<mlir::OpBuilder::InsertPoint>(m, "InsertPoint");

  py::class_<mlir::OpBuilder>(m, "builder", py::dynamic_attr())
      .def(py::init<mlir::MLIRContext *>())
      // // getters
      .def_property_readonly("context", &mlir::OpBuilder::getContext,
                             ret::reference)
      .def("create_module",
           [](mlir::OpBuilder &self) -> mlir::ModuleOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::ModuleOp>(loc);
           })
      .def("ret",
           [](mlir::OpBuilder &self, std::vector<mlir::Value> &vals) -> void {
             auto loc = self.getUnknownLoc();
             self.create<mlir::ReturnOp>(loc, vals);
           })
      .def("call",
           [](mlir::OpBuilder &self, mlir::FuncOp &func,
              std::vector<mlir::Value> &args) -> mlir::OpState {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::CallOp>(loc, func, args);
           })
      // insertion block/point
      .def("set_insertion_point_to_start",
           [](mlir::OpBuilder &self, mlir::Block &block) -> void {
             self.setInsertionPointToStart(&block);
           })
      .def("set_insertion_point_to_end",
           [](mlir::OpBuilder &self, mlir::Block &block) {
             self.setInsertionPointToEnd(&block);
           })
      .def("set_insertion_point_after",
           [](mlir::OpBuilder &self, mlir::Operation &op) {
             self.setInsertionPointAfter(&op);
           })
      .def(
          "get_insertion_block",
          [](mlir::OpBuilder &self) -> mlir::Block * {
            return self.getInsertionBlock();
          },
          ret::reference)
      .def("get_insertion_point", &mlir::OpBuilder::saveInsertionPoint)
      .def("restore_insertion_point", &mlir::OpBuilder::restoreInsertionPoint)
      // .def("set_insert_point", [](ir::builder *self,
      // std::pair<ir::basic_block*, ir::instruction*> pt) {
      //   ir::basic_block *bb = pt.first;
      //   ir::instruction *instr = pt.second;
      //   if (instr) {
      //     if (bb != instr->get_parent())
      //       throw std::runtime_error("invalid insertion point, instr not in
      //       bb");
      //     self->set_insert_point(instr);
      //   } else {
      //     assert(bb);
      //     self->set_insert_point(bb);
      //   }
      // })
      // Attr
      .def("get_bool_attr", &mlir::OpBuilder::getBoolAttr)
      .def("get_int32_attr", &mlir::OpBuilder::getI32IntegerAttr)
      // Use arith.ConstantOp to create constants
      // Constants
      .def("get_int1",
           [](mlir::OpBuilder &self, bool v) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 loc, v, self.getI1Type()));
           })
      .def("get_int8",
           [](mlir::OpBuilder &self, int64_t v) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 loc, v, self.getI8Type()));
           })
      .def("get_int32",
           [](mlir::OpBuilder &self, int64_t v) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 loc, v, self.getI32Type()));
           })
      .def("get_int64",
           [](mlir::OpBuilder &self, int64_t v) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 loc, v, self.getI64Type()));
           })
      // bfloat16 cannot be initialized as it is treated as int16 for now
      //.def("get_bf16",
      //     [](mlir::OpBuilder &self, float v) -> mlir::Value {
      //       auto loc = self.getUnknownLoc();
      //       auto type = self.getBF16Type();
      //       return self.create<mlir::arith::ConstantFloatOp>(
      //           loc,
      //           mlir::APFloat(type.getFloatSemantics(), std::to_string(v)),
      //           type);
      //     })
      .def("get_fp16",
           [](mlir::OpBuilder &self, float v) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::ConstantOp>(
                 loc, self.getF16FloatAttr(v));
           })
      .def("get_fp32",
           [](mlir::OpBuilder &self, float v) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::ConstantOp>(
                 loc, self.getF32FloatAttr(v));
           })
      .def("get_null_value",
           [](mlir::OpBuilder &self, mlir::Type type) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             if (auto floatTy = type.dyn_cast<mlir::FloatType>())
               return self.create<mlir::arith::ConstantFloatOp>(
                   loc, mlir::APFloat(floatTy.getFloatSemantics(), 0), floatTy);
             else if (auto intTy = type.dyn_cast<mlir::IntegerType>())
               return self.create<mlir::arith::ConstantIntOp>(loc, 0, intTy);
             else
               throw std::runtime_error("Not implemented");
           })
      .def("get_all_ones_value",
           [](mlir::OpBuilder &self, mlir::Type type) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             uint64_t val = 0xFFFFFFFFFFFFFFFF;
             if (auto intTy = type.dyn_cast<mlir::IntegerType>())
               return self.create<mlir::arith::ConstantIntOp>(loc, val, intTy);
             else
               throw std::runtime_error("Not implemented");
           })

      // Types
      .def("get_void_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getNoneType();
           })
      .def("get_int1_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getI1Type();
           }) // or ret::copy?
      .def("get_int8_ty",
           [](mlir::OpBuilder &self) -> mlir::Type { return self.getI8Type(); })
      .def("get_int16_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getType<mlir::IntegerType>(16);
           })
      .def(
          "get_int32_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getI32Type(); })
      .def(
          "get_int64_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getI64Type(); })
      .def("get_fp8_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getType<mlir::triton::Float8Type>();
           })
      .def(
          "get_half_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getF16Type(); })
      .def("get_bf16_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getBF16Type();
           })
      .def(
          "get_float_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getF32Type(); })
      .def(
          "get_double_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getF64Type(); })
      .def("get_ptr_ty",
           [](mlir::OpBuilder &self, mlir::Type &type,
              int addrSpace) -> mlir::Type {
             return mlir::triton::PointerType::get(type, addrSpace);
           })
      .def("get_block_ty",
           [](mlir::OpBuilder &self, mlir::Type &elementType,
              std::vector<int64_t> &shape) -> mlir::Type {
             return mlir::RankedTensorType::get(shape, elementType);
           })
      .def("get_function_ty",
           [](mlir::OpBuilder &self, std::vector<mlir::Type> inTypes,
              std::vector<mlir::Type> outTypes) -> mlir::Type {
             return self.getFunctionType(inTypes, outTypes);
           })

      // Ops
      .def("get_or_insert_function",
           [](mlir::OpBuilder &self, mlir::ModuleOp &module,
              std::string &funcName, mlir::Type &funcType,
              std::string &visibility) -> mlir::FuncOp {
             if (mlir::Operation *funcOperation = module.lookupSymbol(funcName))
               return llvm::dyn_cast<mlir::FuncOp>(funcOperation);
             auto loc = self.getUnknownLoc();
             if (auto funcTy = funcType.dyn_cast<mlir::FunctionType>()) {
               llvm::SmallVector<mlir::NamedAttribute> attrs = {
                   mlir::NamedAttribute(self.getStringAttr("sym_visibility"),
                                        self.getStringAttr(visibility))};
               return self.create<mlir::FuncOp>(loc, funcName, funcTy, attrs);
             }
             throw std::runtime_error("invalid function type");
           })
      .def(
          "create_block",
          [](mlir::OpBuilder &self) -> mlir::Block * {
            mlir::Region *parent = self.getBlock()->getParent();
            return self.createBlock(parent);
          },
          ret::reference)
      .def(
          "create_block_with_parent",
          [](mlir::OpBuilder &self, mlir::Region &parent,
             std::vector<mlir::Type> &argTypes) -> mlir::Block * {
            auto argLoc = self.getUnknownLoc();
            llvm::SmallVector<mlir::Location, 8> argLocs(argTypes.size(),
                                                         argLoc);
            return self.createBlock(&parent, {}, argTypes, argLocs);
          },
          ret::reference)
      .def(
          "new_block",
          [](mlir::OpBuilder &self) -> mlir::Block * {
            return new mlir::Block();
          },
          ret::reference)
      // Unstructured control flow
      .def("create_cond_branch",
           [](mlir::OpBuilder &self, mlir::Value condition,
              mlir::Block *trueDest, mlir::Block *falseDest) {
             auto loc = self.getUnknownLoc();
             self.create<mlir::CondBranchOp>(loc, condition, trueDest,
                                             falseDest);
             return;
           })
      .def("create_branch",
           [](mlir::OpBuilder &self, mlir::Block *dest,
              std::vector<mlir::Value> &args) {
             auto loc = self.getUnknownLoc();
             self.create<mlir::BranchOp>(loc, dest, args);
             return;
           })
      // Structured control flow
      .def("create_for_op",
           [](mlir::OpBuilder &self, mlir::Value &lb, mlir::Value &ub,
              mlir::Value &step,
              std::vector<mlir::Value> &initArgs) -> mlir::scf::ForOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::ForOp>(loc, lb, ub, step, initArgs);
           })
      .def("create_if_op",
           [](mlir::OpBuilder &self, std::vector<mlir::Type> &retTypes,
              mlir::Value &condition, bool withElse) -> mlir::scf::IfOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::IfOp>(loc, retTypes, condition,
                                                 withElse);
           })
      .def("create_yield_op",
           [](mlir::OpBuilder &self,
              std::vector<mlir::Value> &yields) -> mlir::scf::YieldOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::YieldOp>(loc, yields);
           })
      .def("create_while_op",
           [](mlir::OpBuilder &self, std::vector<mlir::Type> &retTypes,
              std::vector<mlir::Value> &initArgs) -> mlir::scf::WhileOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::WhileOp>(loc, retTypes, initArgs);
           })
      .def("create_condition_op",
           [](mlir::OpBuilder &self, mlir::Value &cond,
              std::vector<mlir::Value> &args) -> mlir::scf::ConditionOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::ConditionOp>(loc, cond, args);
           })

      // miscellaneous
      .def("create_make_range",
           [](mlir::OpBuilder &self, int start, int end) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             auto retType =
                 mlir::RankedTensorType::get({end - start}, self.getI32Type());
             return self.create<mlir::triton::MakeRangeOp>(loc, retType, start,
                                                           end);
           })

      // Cast instructions
      // Conversions for custom FP types (FP8)
      .def("create_fp_to_fp",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::FpToFpOp>(loc, dstType, src);
           })
      // Conversions for standard LLVM builtin types
      .def("create_bitcast",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::BitcastOp>(loc, dstType, src);
           })
      .def("create_si_to_fp",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::SIToFPOp>(loc, dstType, src);
           })
      .def("create_ui_to_fp",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::UIToFPOp>(loc, dstType, src);
           })
      .def("create_fp_to_si",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::FPToSIOp>(loc, dstType, src);
           })
      .def("create_fp_to_ui",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::FPToUIOp>(loc, dstType, src);
           })
      .def("create_fp_ext",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::ExtFOp>(loc, dstType, src);
           })
      .def("create_fp_trunc",
           [](mlir::OpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::TruncFOp>(loc, dstType, src);
           })
      .def("create_int_cast",
           [](mlir::OpBuilder &self, mlir::Value &src, mlir::Type &dstType,
              bool isSigned) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             // get element type if necessary
             mlir::Type srcType = src.getType();
             auto srcTensorType = srcType.dyn_cast<mlir::RankedTensorType>();
             auto dstTensorType = dstType.dyn_cast<mlir::RankedTensorType>();
             mlir::Type srcEltType = srcType;
             mlir::Type dstEltType = dstType;
             if (dstTensorType && srcTensorType) {
               dstEltType = dstTensorType.getElementType();
               srcEltType = srcTensorType.getElementType();
             }
             unsigned srcWidth = srcEltType.getIntOrFloatBitWidth();
             unsigned dstWidth = dstEltType.getIntOrFloatBitWidth();
             if (srcWidth == dstWidth)
               return self.create<mlir::arith::BitcastOp>(loc, dstType, src);
             else if (srcWidth > dstWidth)
               return self.create<mlir::arith::TruncIOp>(loc, dstType, src);
             else if (isSigned)
               return self.create<mlir::arith::ExtSIOp>(loc, dstType, src);
             else
               return self.create<mlir::arith::ExtUIOp>(loc, dstType, src);
           })
      .def("create_to_index",
           [](mlir::OpBuilder &self, mlir::Value &input) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::IndexCastOp>(loc, input,
                                                          self.getIndexType());
           })
      .def("create_index_to_si",
           [](mlir::OpBuilder &self, mlir::Value &input) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::IndexCastOp>(loc, input,
                                                          self.getI32Type());
           })
      .def("create_fmul",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::MulFOp>(loc, lhs, rhs);
           })
      .def("create_fdiv",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::DivFOp>(loc, lhs, rhs);
           })
      .def("create_frem",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::RemFOp>(loc, lhs, rhs);
           })
      .def("create_fadd",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::AddFOp>(loc, lhs, rhs);
           })
      .def("create_fsub",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::SubFOp>(loc, lhs, rhs);
           })
      .def("create_mul",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::MulIOp>(loc, lhs, rhs);
           })
      .def("create_sdiv",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
           })
      .def("create_udiv",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::DivUIOp>(loc, lhs, rhs);
           })
      .def("create_srem",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
           })
      .def("create_urem",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::RemUIOp>(loc, lhs, rhs);
           })
      .def("create_add",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::AddIOp>(loc, lhs, rhs);
           })
      .def("create_sub",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(
                 self.create<mlir::arith::SubIOp>(loc, lhs, rhs));
           })
      .def("create_shl",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(
                 self.create<mlir::arith::ShLIOp>(loc, lhs, rhs));
           })
      .def("create_lshr",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(
                 self.create<mlir::arith::ShRUIOp>(loc, lhs, rhs));
           })
      .def("create_ashr",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(
                 self.create<mlir::arith::ShRSIOp>(loc, lhs, rhs));
           })
      // AddPtr (similar to GEP)
      .def("create_addptr",
           [](mlir::OpBuilder &self, mlir::Value &ptr,
              mlir::Value &offset) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::AddPtrOp>(loc, ptr.getType(), ptr,
                                                        offset);
           })
      // Comparison (int)
      .def("create_icmpSLE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::sle, lhs, rhs);
           })
      .def("create_icmpSLT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::slt, lhs, rhs);
           })
      .def("create_icmpSGE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::sge, lhs, rhs);
           })
      .def("create_icmpSGT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::sgt, lhs, rhs);
           })
      .def("create_icmpULE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::ule, lhs, rhs);
           })
      .def("create_icmpULT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::ult, lhs, rhs);
           })
      .def("create_icmpUGE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::uge, lhs, rhs);
           })
      .def("create_icmpUGT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::ugt, lhs, rhs);
           })
      .def("create_icmpEQ",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
           })
      .def("create_icmpNE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpIOp>(
                 loc, mlir::arith::CmpIPredicate::ne, lhs, rhs);
           })
      // Comparison (float)
      .def("create_fcmpOLT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::OLT, lhs, rhs);
           })
      .def("create_fcmpOGT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
           })
      .def("create_fcmpOLE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::OLE, lhs, rhs);
           })
      .def("create_fcmpOGE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::OGE, lhs, rhs);
           })
      .def("create_fcmpOEQ",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
           })
      .def("create_fcmpONE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::ONE, lhs, rhs);
           })
      .def("create_fcmpULT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::ULT, lhs, rhs);
           })
      .def("create_fcmpUGT",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::UGT, lhs, rhs);
           })
      .def("create_fcmpULE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::ULE, lhs, rhs);
           })
      .def("create_fcmpUGE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::UGE, lhs, rhs);
           })
      .def("create_fcmpUEQ",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::UEQ, lhs, rhs);
           })
      .def("create_fcmpUNE",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::CmpFOp>(
                 loc, mlir::arith::CmpFPredicate::UNE, lhs, rhs);
           })
      // // Logical
      .def("create_and",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::AndIOp>(loc, lhs, rhs);
           })
      .def("create_xor",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
           })
      .def("create_or",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::OrIOp>(loc, lhs, rhs);
           })
      // Input/Output
      .def("create_load",
           [](mlir::OpBuilder &self, mlir::Value &ptrs,
              mlir::triton::CacheModifier cacheModifier,
              mlir::triton::EvictionPolicy evictionPolicy,
              bool isVolatile) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::LoadOp>(
                 loc, ptrs, cacheModifier, evictionPolicy, isVolatile);
           })
      .def("create_store",
           [](mlir::OpBuilder &self, mlir::Value &ptrs,
              mlir::Value &value) -> void {
             auto loc = self.getUnknownLoc();
             self.create<mlir::triton::StoreOp>(loc, ptrs, value);
           })
      .def("create_masked_load",
           [](mlir::OpBuilder &self, mlir::Value &ptrs, mlir::Value &mask,
              std::optional<mlir::Value> &other,
              mlir::triton::CacheModifier cacheModifier,
              mlir::triton::EvictionPolicy evictionPolicy,
              bool isVolatile) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::LoadOp>(
                 loc, ptrs, mask, other.value_or(mlir::Value()), cacheModifier,
                 evictionPolicy, isVolatile);
           })
      .def("create_masked_store",
           [](mlir::OpBuilder &self, mlir::Value &ptrs, mlir::Value &val,
              mlir::Value &mask) -> void {
             auto loc = self.getUnknownLoc();
             self.create<mlir::triton::StoreOp>(loc, ptrs, val, mask);
           })
      .def("create_view",
           [](mlir::OpBuilder &self, mlir::Value &arg,
              std::vector<int64_t> &shape) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             auto argType = arg.getType()
                                .dyn_cast<mlir::RankedTensorType>()
                                .getElementType();
             return self.create<mlir::triton::ViewOp>(
                 loc, mlir::RankedTensorType::get(shape, argType), arg);
           })
      .def(
          "create_expand_dims",
          [](mlir::OpBuilder &self, mlir::Value &arg, int axis) -> mlir::Value {
            auto loc = self.getUnknownLoc();
            auto argType = arg.getType().dyn_cast<mlir::RankedTensorType>();
            auto argEltType = argType.getElementType();
            std::vector<int64_t> retShape = argType.getShape();
            retShape.insert(retShape.begin() + axis, 1);
            return self.create<mlir::triton::ExpandDimsOp>(
                loc, mlir::RankedTensorType::get(retShape, argEltType), arg,
                axis);
          })
      .def("create_cat",
           [](mlir::OpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             auto lhsType = lhs.getType().dyn_cast<mlir::RankedTensorType>();
             auto rhsType = rhs.getType().dyn_cast<mlir::RankedTensorType>();
             if (!(lhsType.getShape().size() == 1 &&
                   rhsType.getShape().size() == 1))
               throw std::runtime_error(
                   "shape not supported by cat. Expecting rank-1 inputs");
             std::vector<int64_t> shape{lhsType.getShape()[0] +
                                        rhsType.getShape()[0]};
             return self.create<mlir::triton::CatOp>(
                 loc,
                 mlir::RankedTensorType::get(shape, lhsType.getElementType()),
                 lhs, rhs);
           })
      .def("create_trans",
           [](mlir::OpBuilder &self, mlir::Value &arg) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             auto argType = arg.getType().dyn_cast<mlir::RankedTensorType>();
             auto argEltType = argType.getElementType();
             std::vector<int64_t> retShape = argType.getShape();
             std::reverse(retShape.begin(), retShape.end());
             return self.create<mlir::triton::TransOp>(
                 loc, mlir::RankedTensorType::get(retShape, argEltType), arg);
           })
      .def("create_broadcast",
           [](mlir::OpBuilder &self, mlir::Value &arg,
              std::vector<int64_t> &shape) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             if (auto argType =
                     arg.getType().dyn_cast<mlir::RankedTensorType>())
               return self.createOrFold<mlir::triton::BroadcastOp>(
                   loc,
                   mlir::RankedTensorType::get(shape, argType.getElementType()),
                   arg);
             throw std::runtime_error(
                 "arg is not of RankedTensorType, use create_splat");
           })
      .def("create_splat",
           [](mlir::OpBuilder &self, mlir::Value &arg,
              std::vector<int64_t> &shape) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             auto argType = arg.getType();
             auto ret = self.createOrFold<mlir::triton::SplatOp>(
                 loc, mlir::RankedTensorType::get(shape, argType), arg);
             return ret;
           })
      // // atomic
      .def("create_atomic_cas",
           [](mlir::OpBuilder &self, mlir::Value &ptr, mlir::Value &cmp,
              mlir::Value &val) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             mlir::Type dstType;
             if (auto srcTensorType =
                     ptr.getType().dyn_cast<mlir::RankedTensorType>()) {
               mlir::Type dstElemType = srcTensorType.getElementType()
                                            .cast<mlir::triton::PointerType>()
                                            .getPointeeType();
               dstType = mlir::RankedTensorType::get(srcTensorType.getShape(),
                                                     dstElemType);
             } else {
               auto ptrType = mlir::getElementTypeOrSelf(ptr)
                                  .cast<mlir::triton::PointerType>();
               dstType = ptrType.getPointeeType();
             }
             return self.create<mlir::triton::AtomicCASOp>(loc, dstType, ptr,
                                                           cmp, val);
           })
      .def("create_atomic_rmw",
           [](mlir::OpBuilder &self, mlir::triton::RMWOp rmwOp,
              mlir::Value &ptr, mlir::Value &val,
              mlir::Value &mask) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             mlir::Type dstType;
             if (auto srcTensorType =
                     ptr.getType().dyn_cast<mlir::RankedTensorType>()) {
               mlir::Type dstElemType = srcTensorType.getElementType()
                                            .cast<mlir::triton::PointerType>()
                                            .getPointeeType();
               dstType = mlir::RankedTensorType::get(srcTensorType.getShape(),
                                                     dstElemType);
             } else {
               auto ptrType = mlir::getElementTypeOrSelf(ptr)
                                  .cast<mlir::triton::PointerType>();
               dstType = ptrType.getPointeeType();
             }
             return self.create<mlir::triton::AtomicRMWOp>(loc, dstType, rmwOp,
                                                           ptr, val, mask);
           })
      // External
      .def("create_external_elementwise",
           [](mlir::OpBuilder &self, const std::string &libName,
              const std::string &libPath, const std::string &symbol,
              std::vector<mlir::Value> &argList,
              mlir::Type retType) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::ExtElemwiseOp>(
                 loc, retType, argList, libName, libPath, symbol);
           })
      // Built-in instruction
      .def("create_get_program_id",
           [](mlir::OpBuilder &self, int axis) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::GetProgramIdOp>(
                 loc, self.getI32Type(), self.getI32IntegerAttr(axis));
           })
      .def("create_get_num_programs",
           [](mlir::OpBuilder &self, int axis) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::GetNumProgramsOp>(
                 loc, self.getI32Type(), self.getI32IntegerAttr(axis));
           })
      .def("create_dot",
           [](mlir::OpBuilder &self, mlir::Value &a, mlir::Value &b,
              mlir::Value &c, bool allowTF32) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::DotOp>(loc, c.getType(), a, b, c,
                                                     allowTF32);
           })
      .def("create_exp",
           [](mlir::OpBuilder &self, mlir::Value &val) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::math::ExpOp>(loc, val);
           })
      .def("create_cos",
           [](mlir::OpBuilder &self, mlir::Value &val) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::math::CosOp>(loc, val);
           })
      .def("create_sin",
           [](mlir::OpBuilder &self, mlir::Value &val) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::math::SinOp>(loc, val);
           })
      .def("create_log",
           [](mlir::OpBuilder &self, mlir::Value &val) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::math::LogOp>(loc, val);
           })
      .def("create_sqrt",
           [](mlir::OpBuilder &self, mlir::Value &val) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::math::SqrtOp>(loc, val);
           })
      .def("create_reduce",
           [](mlir::OpBuilder &self, mlir::Value &operand,
              mlir::triton::RedOp redOp, int axis) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             auto inputTensorType =
                 operand.getType().dyn_cast<mlir::RankedTensorType>();
             std::vector<int64_t> shape = inputTensorType.getShape();
             shape.erase(shape.begin() + axis);
             bool withIndex = mlir::triton::ReduceOp::withIndex(redOp);
             mlir::Type resType = withIndex ? self.getI32Type()
                                            : inputTensorType.getElementType();
             if (!shape.empty()) {
               resType = mlir::RankedTensorType::get(shape, resType);
             }
             return self.create<mlir::triton::ReduceOp>(loc, resType, redOp,
                                                        operand, axis);
           })
      .def("create_ptr_to_int",
           [](mlir::OpBuilder &self, mlir::Value &val,
              mlir::Type &type) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::PtrToIntOp>(loc, type, val);
           })
      .def("create_int_to_ptr",
           [](mlir::OpBuilder &self, mlir::Value &val,
              mlir::Type &type) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::triton::IntToPtrOp>(loc, type, val);
           })
      .def("create_select",
           [](mlir::OpBuilder &self, mlir::Value &condition,
              mlir::Value &trueValue, mlir::Value &falseValue) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::SelectOp>(loc, condition, trueValue,
                                                falseValue);
           })
      .def("create_printf",
           [](mlir::OpBuilder &self, const std::string &prefix,
              const std::vector<mlir::Value> &values) -> void {
             auto loc = self.getUnknownLoc();
             self.create<mlir::triton::PrintfOp>(
                 loc,
                 mlir::StringAttr::get(self.getContext(),
                                       llvm::StringRef(prefix)),
                 values);
           })
      // Undef
      .def("create_undef",
           [](mlir::OpBuilder &self, mlir::Type &type) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<::mlir::LLVM::UndefOp>(loc, type);
           })
      // Force GPU barrier
      .def("create_barrier", [](mlir::OpBuilder &self) {
        auto loc = self.getUnknownLoc();
        self.create<mlir::gpu::BarrierOp>(loc);
      });

  py::class_<mlir::PassManager>(m, "pass_manager")
      .def(py::init<mlir::MLIRContext *>())
      .def("enable_debug",
           [](mlir::PassManager &self) {
             auto printingFlags = mlir::OpPrintingFlags();
             printingFlags.elideLargeElementsAttrs(16);
             self.enableIRPrinting(
                 /*shouldPrintBeforePass=*/nullptr,
                 /*shouldPrintAfterPass=*/
                 [](mlir::Pass *pass, mlir::Operation *) {
                   return ::triton::tools::getBoolEnv("MLIR_ENABLE_DUMP");
                 },
                 /*printModuleScope=*/false,
                 /*printAfterOnlyOnChange=*/true,
                 /*printAfterOnlyOnFailure*/ false, llvm::dbgs(),
                 printingFlags);
           })
      .def("run",
           [](mlir::PassManager &self, mlir::ModuleOp &mod) {
             // TODO: maybe dump module to file and print error for better
             // diagnostics
             if (mlir::failed(self.run(mod.getOperation())))
               throw std::runtime_error("PassManager::run failed");
           })
      .def(
          "add_sccp_pass",
          [](mlir::PassManager &self) { self.addPass(mlir::createSCCPPass()); })
      .def("add_coalesce_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createTritonGPUCoalescePass());
           })
      .def("add_symbol_dce_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createSymbolDCEPass());
           })
      .def("add_inliner_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createInlinerPass());
           })
      .def("add_canonicalizer_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createCanonicalizerPass());
           })
      .def("add_cse_pass",
           [](mlir::PassManager &self) { self.addPass(mlir::createCSEPass()); })
      .def("add_licm_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createLoopInvariantCodeMotionPass());
           })
      .def("add_triton_combine_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::triton::createCombineOpsPass());
           })
      .def("add_convert_triton_to_tritongpu_pass",
           [](mlir::PassManager &self, int numWarps) {
             self.addPass(
                 mlir::triton::createConvertTritonToTritonGPUPass(numWarps));
           })
      .def("add_tritongpu_pipeline_pass",
           [](mlir::PassManager &self, int numStages) {
             self.addPass(mlir::createTritonGPUPipelinePass(numStages));
           })
      .def("add_tritongpu_prefetch_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createTritonGPUPrefetchPass());
           })
      .def("add_tritongpu_combine_pass",
           [](mlir::PassManager &self, int computeCapability) {
             self.addPass(
                 mlir::createTritonGPUCombineOpsPass(computeCapability));
           })
      .def("add_tritongpu_update_mma_for_volta_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createTritonGPUUpdateMmaForVoltaPass());
           })
      .def("add_tritongpu_reorder_instructions_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createTritonGPUReorderInstructionsPass());
           })
      .def("add_tritongpu_decompose_conversions_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createTritonGPUDecomposeConversionsPass());
           })
      .def("add_triton_gpu_to_llvm",
           [](mlir::PassManager &self) {
             self.addPass(mlir::triton::createConvertTritonGPUToLLVMPass());
           })
      .def("add_scf_to_cfg", [](mlir::PassManager &self) {
        self.addPass(mlir::createLowerToCFGPass());
      });
}

void init_triton_translation(py::module &m) {
  using ret = py::return_value_policy;

  m.def("get_shared_memory_size", [](mlir::ModuleOp mod) {
    auto shared = mod->getAttrOfType<mlir::IntegerAttr>("triton_gpu.shared");
    return shared.getInt();
  });

  m.def(
      "translate_triton_gpu_to_llvmir",
      [](mlir::ModuleOp op, int computeCapability) {
        py::gil_scoped_release allow_threads;
        llvm::LLVMContext llvmContext;
        auto llvmModule = ::mlir::triton::translateTritonGPUToLLVMIR(
            &llvmContext, op, computeCapability);
        if (!llvmModule)
          llvm::report_fatal_error("Failed to translate TritonGPU to LLVM IR.");

        std::string str;
        llvm::raw_string_ostream os(str);
        llvmModule->print(os, nullptr);
        os.flush();
        return str;
      },
      ret::take_ownership);

  m.def(
      "translate_llvmir_to_ptx",
      [](const std::string llvmIR, int capability, int version) -> std::string {
        py::gil_scoped_release allow_threads;
        // create LLVM module from C++
        llvm::LLVMContext context;
        std::unique_ptr<llvm::MemoryBuffer> buffer =
            llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
        llvm::SMDiagnostic error;
        std::unique_ptr<llvm::Module> module =
            llvm::parseIR(buffer->getMemBufferRef(), error, context);
        if (!module) {
          llvm::report_fatal_error(
              "failed to parse IR: " + error.getMessage() +
              "lineno: " + std::to_string(error.getLineNo()));
        }

        // translate module to PTX
        auto ptxCode =
            triton::translateLLVMIRToPTX(*module, capability, version);
        return ptxCode;
      },
      ret::take_ownership);

  m.def("compile_ptx_to_cubin",
        [](const std::string &ptxCode, const std::string &ptxasPath,
           int capability) -> py::object {
          py::gil_scoped_release allow_threads;

          // compile ptx with ptxas
          llvm::SmallString<64> fsrc;
          llvm::SmallString<64> flog;
          llvm::sys::fs::createTemporaryFile("compile-ptx-src", "", fsrc);
          llvm::sys::fs::createTemporaryFile("compile-ptx-log", "", flog);
          std::string fbin = std::string(fsrc) + ".o";
          llvm::FileRemover srcRemover(fsrc);
          llvm::FileRemover logRemover(flog);
          llvm::FileRemover binRemover(fbin);
          const char *_fsrc = fsrc.c_str();
          const char *_flog = flog.c_str();
          const char *_fbin = fbin.c_str();
          std::ofstream ofs(_fsrc);
          ofs << ptxCode << std::endl;
          ofs.close();
          std::string cmd;
          int err;
          cmd = ptxasPath + " -v --gpu-name=sm_" + std::to_string(capability) +
                " " + _fsrc + " -o " + _fsrc + ".o 2> " + _flog;
          err = system(cmd.c_str());
          if (err != 0) {
            std::ifstream _log(_flog);
            std::string log(std::istreambuf_iterator<char>(_log), {});
            throw std::runtime_error("Internal Triton PTX codegen error: \n" +
                                     log);
          }
          std::ifstream _cubin(_fbin, std::ios::binary);
          std::string cubin(std::istreambuf_iterator<char>(_cubin), {});
          _cubin.close();
          py::bytes bytes(cubin);
          return std::move(bytes);
        });

  m.def("add_external_libs",
        [](mlir::ModuleOp &op, const std::vector<std::string> &names,
           const std::vector<std::string> &paths) {
          ::mlir::triton::addExternalLibs(op, names, paths);
        });
}

void init_triton(py::module &m) {
  py::module subm = m.def_submodule("triton");
  // init_triton_codegen(subm.def_submodule("code_gen"));
  init_triton_runtime(subm.def_submodule("runtime"));
  init_triton_ir(subm.def_submodule("ir"));
  init_triton_translation(subm);
}
