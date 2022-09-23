#ifndef TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ASM_FORMAT_H_
#define TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ASM_FORMAT_H_

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir {
class ConversionPatternRewriter;
class Location;

namespace triton {
using llvm::StringRef;

class PTXInstr;
class PTXInstrCommon;
class PTXInstrExecution;

// PTXBuilder helps to manage a PTX asm program consists of one or multiple
// instructions.
//
// A helper for building a ASM program, the objective of PTXBuilder is to give a
// thin encapsulation and make the ASM code for MLIR LLVM Dialect more clear.
// Currently, several factors are introduced to reduce the need for mixing
// string and C++ if-else code.
//
// Usage:
// To build: @$3 asm("@%3 add.s32 %0, %1, %2;" : "=r"(i) : "r"(j), "r"(k),
// "b"(p));
//
// PTXBuilder builder;
// auto& add = builder.create<>();
// add.predicate(pVal).o("lo").o("u32"); // add any suffix
// // predicate here binds %0 to pVal, pVal is a mlir::Value
//
// auto* iOpr = builder.newOperand(iVal, "r"); // %1 bind to iVal
// auto* jOpr = builder.newOperand(jVal, "r"); // %2 bind to jVal
// auto* kOpr = builder.newOperand(kVal, "r"); // %3 bind to kVal
// add(iOpr, jOpr, kOpr).predicate(predVal);   // set operands and predicate
//
// To get the asm code:
// builder.dump()
//
// To get all the mlir::Value used in the PTX code,
//
// builder.getAllMlirArgs() // get {pVal, iVal, jVal, kVal}
//
// To get the string containing all the constraints with "," separated,
// builder.getConstraints() // get "=r,r,k"
//
// PTXBuilder can build a PTX asm with multiple instructions, sample code:
//
// PTXBuilder builder;
// auto& mov = builder.create("mov");
// auto& cp = builder.create("cp");
// mov(...);
// cp(...);
// This will get a PTX code with two instructions.
//
// Similar to a C function, a declared PTXInstr instance can be launched
// multiple times with different operands, e.g.
//
//   auto& mov = builder.create("mov");
//   mov(... some operands ...);
//   mov(... some different operands ...);
//
// Finally, we will get a PTX code with two mov instructions.
//
// There are several derived instruction type for typical instructions, for
// example, the PtxIOInstr for ld and st instructions.
struct PTXBuilder {
  struct Operand {
    std::string constraint;
    Value value;
    int idx{-1};
    llvm::SmallVector<Operand *> list;
    std::function<std::string(int idx)> repr;

    // for list
    Operand() = default;
    Operand(const Operation &) = delete;
    Operand(Value value, StringRef constraint)
        : value(value), constraint(constraint) {}

    bool isList() const { return !value && constraint.empty(); }

    Operand *listAppend(Operand *arg) {
      list.push_back(arg);
      return this;
    }

    Operand *listGet(size_t nth) const {
      assert(nth < list.size());
      return list[nth];
    }

    std::string dump() const;
  };

  template <typename INSTR = PTXInstr> INSTR *create(const std::string &name) {
    instrs.emplace_back(std::make_unique<INSTR>(this, name));
    return static_cast<INSTR *>(instrs.back().get());
  }

  // Create a list of operands.
  Operand *newListOperand() { return newOperand(); }

  Operand *newListOperand(ArrayRef<std::pair<mlir::Value, std::string>> items) {
    auto *list = newOperand();
    for (auto &item : items) {
      list->listAppend(newOperand(item.first, item.second));
    }
    return list;
  }

  Operand *newListOperand(unsigned count, mlir::Value val,
                          const std::string &constraint) {
    auto *list = newOperand();
    for (int i = 0; i < count; i++) {
      list->listAppend(newOperand(val, constraint));
    }
    return list;
  }

  Operand *newListOperand(unsigned count, const std::string &constraint) {
    auto *list = newOperand();
    for (int i = 0; i < count; i++) {
      list->listAppend(newOperand(constraint));
    }
    return list;
  }

  // Create a new operand. It will not add to operand list.
  // @value: the MLIR value bind to this operand.
  // @constraint: ASM operand constraint, .e.g. "=r"
  // @formatter: extra format to represent this operand in ASM code, default is
  //             "%{0}".format(operand.idx).
  Operand *newOperand(mlir::Value value, StringRef constraint,
                      std::function<std::string(int idx)> formatter = nullptr);

  // Create a new operand which is written to, that is, the constraint starts
  // with "=", e.g. "=r".
  Operand *newOperand(StringRef constraint);

  // Create a constant integer operand.
  Operand *newConstantOperand(int v);
  // Create a constant operand with explicit code specified.
  Operand *newConstantOperand(const std::string &v);

  Operand *newAddrOperand(mlir::Value addr, StringRef constraint, int off = 0);

  llvm::SmallVector<Operand *, 4> getAllArgs() const;

  llvm::SmallVector<Value, 4> getAllMLIRArgs() const;

  std::string getConstraints() const;

  std::string dump() const;

  mlir::Value launch(ConversionPatternRewriter &rewriter, Location loc,
                     Type resTy, bool hasSideEffect = true,
                     bool isAlignStack = false,
                     ArrayRef<Attribute> attrs = {}) const;

private:
  Operand *newOperand() {
    argArchive.emplace_back(std::make_unique<Operand>());
    return argArchive.back().get();
  }

  friend class PTXInstr;
  friend class PTXInstrCommon;

protected:
  llvm::SmallVector<std::unique_ptr<Operand>, 6> argArchive;
  llvm::SmallVector<std::unique_ptr<PTXInstrCommon>, 2> instrs;
  llvm::SmallVector<std::unique_ptr<PTXInstrExecution>, 4> executions;
  int oprCounter{};
};

// PTX instruction common interface.
// Put the generic logic for all the instructions here.
struct PTXInstrCommon {
  explicit PTXInstrCommon(PTXBuilder *builder) : builder(builder) {}

  using Operand = PTXBuilder::Operand;

  // clang-format off
  PTXInstrExecution& operator()(Operand* a) { return call({a}); }
  PTXInstrExecution& operator()(Operand* a, Operand* b) { return call({a, b}); }
  PTXInstrExecution& operator()(Operand* a, Operand* b, Operand* c) { return call({a, b, c}); }
  PTXInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d) { return call({a, b, c, d}); }
  PTXInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e) { return call({a, b, c, d, e}); }
  PTXInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e, Operand* f) { return call({a, b, c, d, e, f}); }
  PTXInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e, Operand* f, Operand* g) { return call({a, b, c, d, e, f, g}); }
  // clang-format on

  // Set operands of this instruction.
  PTXInstrExecution &operator()(llvm::ArrayRef<Operand *> oprs);

protected:
  PTXInstrExecution &call(llvm::ArrayRef<Operand *> oprs);

  PTXBuilder *builder{};
  llvm::SmallVector<std::string, 4> instrParts;

  friend class PTXInstrExecution;
};

template <class ConcreteT> struct PTXInstrBase : public PTXInstrCommon {
  using Operand = PTXBuilder::Operand;

  explicit PTXInstrBase(PTXBuilder *builder, const std::string &name)
      : PTXInstrCommon(builder) {
    o(name);
  }

  // Append a suffix to the instruction.
  // e.g. PTXInstr("add").o("s32") get a add.s32.
  // A predicate is used to tell whether to apply the suffix, so that no if-else
  // code needed. e.g. `PTXInstr("add").o("s32", isS32).o("u32", !isS32);` will
  // get a `add.s32` if isS32 is true.
  ConcreteT &o(const std::string &suffix, bool predicate = true) {
    if (predicate)
      instrParts.push_back(suffix);
    return *static_cast<ConcreteT *>(this);
  }
};

struct PTXInstr : public PTXInstrBase<PTXInstr> {
  using PTXInstrBase<PTXInstr>::PTXInstrBase;
};

// A helper for PTX ld/st instruction.
// Usage:
// PtxIOInstr store("st");
// store.predicate(pValue).global().v(32).b(1); // @%0 st.global.v32.b1
// store.addAddr(addrValue, "l", off);
struct PtxIOInstr : public PTXInstrBase<PtxIOInstr> {
  using PTXInstrBase<PtxIOInstr>::PTXInstrBase;

  // Add ".global" suffix to instruction
  PtxIOInstr &global(bool predicate = true) {
    o("global", predicate);
    return *this;
  }

  // Add ".v" suffix to instruction
  PtxIOInstr &v(int vecWidth, bool predicate = true) {
    if (vecWidth > 1) {
      o("v" + std::to_string(vecWidth), predicate);
    }
    return *this;
  }

  // Add ".b" suffix to instruction
  PtxIOInstr &b(int width) {
    o("b" + std::to_string(width));
    return *this;
  }
};

// Record the operands and context for "launching" a PtxInstr.
struct PTXInstrExecution {
  using Operand = PTXBuilder::Operand;

  llvm::SmallVector<Operand *> argsInOrder;

  PTXInstrExecution() = default;
  explicit PTXInstrExecution(PTXInstrCommon *instr,
                             llvm::ArrayRef<Operand *> oprs)
      : instr(instr), argsInOrder(oprs.begin(), oprs.end()) {}

  // Prefix a predicate to the instruction.
  PTXInstrExecution &predicate(mlir::Value value, StringRef constraint = "b") {
    pred = instr->builder->newOperand(value, constraint);
    return *this;
  }

  // Prefix a !predicate to the instruction.
  PTXInstrExecution &predicateNot(mlir::Value value, StringRef constraint) {
    pred = instr->builder->newOperand(value, constraint);
    pred->repr = [](int idx) { return "@!%" + std::to_string(idx); };
    return *this;
  }

  std::string dump() const;

  SmallVector<Operand *> getArgList() const;

  PTXInstrCommon *instr{};
  Operand *pred{};
};

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ASM_FORMAT_H_
