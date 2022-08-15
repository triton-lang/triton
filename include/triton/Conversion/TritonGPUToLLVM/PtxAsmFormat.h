#ifndef TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ASM_FORMAT_H_
#define TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ASM_FORMAT_H_

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include <memory>
#include <string>

namespace mlir {
namespace triton {
using llvm::StringRef;

// TODO(Superjomn) Move to a global utility file?
std::string strJoin(llvm::ArrayRef<std::string> strs,
                    llvm::StringRef delimiter);

// A helper for building a single inline ASM instruction, the objective of
// PtxInstr is to give a thin encapsulation and make the ASM code for MLIR LLVM
// Dialect more clear. Currently, several factors are introduced to reduce the
// need for mixing string and C++ if-else code.
// Usage:
// To build: asm("add.s32 %0, %1, %2;" : "=r"(i) : "r"(j), "r"(k));
//
// PtxInstr mulr("mul");
// mulr.o("lo").o("u32").addOperand(valueI, "=r") // %0 bind to valueI
//                      .addOperand(valueJ, "r")  // %1 bind to valueJ
//                      .addOperand(valueK, "k"); // %2 bind to valueK
//
// mulr.getConstrains() // get "=r,r,k"
// mulr.getAllMlirArgs() // get {valueI, valueJ, valueK}
//
// TODO(Superjomn) Add multi-line ASM code support and register support later.
struct PtxInstr {
  explicit PtxInstr(const std::string &name) { o(name); }

  struct Operand {
    std::string constraint;
    Value value;
    int idx{-1};
    llvm::SmallVector<Operand *> list;
    std::function<std::string(int idx)> repr;

    // for list
    Operand() = default;
    Operand(Value value, StringRef constraint)
        : value(value), constraint(constraint) {}

    bool isList() const { return !value; }

    Operand *listAppend(Operand *arg) {
      list.push_back(arg);
      return this;
    }

    std::string dump() const;
  };

  // Create a new operand. It will not add to operand list.
  // @value: the MLIR value bind to this operand.
  // @constraint: ASM operand constraint, .e.g. "=r"
  // @formater: extra format to represent this operand in ASM code, default is
  //            "%{0}".format(operand.idx).
  Operand *newOperand(mlir::Value value, StringRef constraint,
                      std::function<std::string(int idx)> formater = nullptr);

  // Append the operand to the intruction's operand list.
  Operand *addOperand(Operand *opr) {
    assert(std::find(argsInOrder.begin(), argsInOrder.end(), opr) ==
           argsInOrder.end());
    argsInOrder.push_back(opr);
    return opr;
  }

  // Create and add an operand to the intruction's operand list.
  Operand *addOperand(mlir::Value value, StringRef constraint) {
    auto *opr = newOperand(value, constraint);
    return addOperand(opr);
  }

  // Prefix a predicate to the instruction.
  PtxInstr &predicate(mlir::Value value, StringRef constraint) {
    pred = newOperand(value, constraint);
    return *this;
  }

  // Append a suffix to the instruction.
  // e.g. PtxInstr("add").o("s32") get a add.s32.
  // A predicate is used to tell whether to apply the suffix, so that no if-else
  // code needed. e.g. `PtxInstr("add").o("s32", isS32).o("u32", !isS32);` will
  // get a `add.s32` if isS32 is true.
  PtxInstr &o(const std::string &suffix, bool predicate = true) {
    if (predicate)
      instrParts.push_back(suffix);
    return *this;
  }

  PtxInstr &addListOperation(llvm::ArrayRef<Operand *> list) {
    auto *opr = newList();
    for (auto *v : list)
      opr->listAppend(v);
    addOperand(opr);
    return *this;
  }

  // Create a list of operands.
  Operand *newList() {
    argArchive.emplace_back(std::make_unique<Operand>());
    return argArchive.back().get();
  }

  std::string dump() const;

  llvm::SmallVector<Operand *, 4> getArgList() const;
  llvm::SmallVector<Operand *, 4> getAllArgs() const {
    llvm::SmallVector<Operand *, 4> res;
    for (auto &x : argArchive)
      if (!x->isList())
        res.push_back(x.get());
    return res;
  }

  std::string getConstrains() const {
    auto args = getAllArgs();
    llvm::SmallVector<std::string, 4> argReprs;
    for (auto arg : args)
      argReprs.push_back(arg->constraint);
    return strJoin(argReprs, ",");
  }

  llvm::SmallVector<Value, 4> getAllMlirArgs() const {
    llvm::SmallVector<Value, 4> res;
    for (auto &arg : argArchive) {
      if (!arg->isList())
        res.push_back(arg->value);
    }
    return res;
  }

protected:
  Operand *pred{};
  int oprCounter{};
  llvm::SmallVector<std::string, 4> instrParts;
  llvm::SmallVector<std::unique_ptr<Operand>, 6> argArchive;
  llvm::SmallVector<Operand *> argsInOrder;
  std::string argStr;
};

// A helper for PTX ld/st instruction.
// Usage:
// PtxIOInstr store("st");
// store.predicate(pValue).global().v(32).b(1); // @%0 st.global.v32.b1
// store.addAddr(addrValue, "l", off);
struct PtxIOInstr : public PtxInstr {
  PtxIOInstr(const std::string &name) : PtxInstr(name) {}

  // Add ".global" suffix to instruction
  PtxIOInstr &global(bool predicate = true) {
    o("global", predicate);
    return *this;
  }

  // Add ".v" suffix to instruction
  PtxIOInstr &v(int vecWidth, bool predicate = true) {
    if (vecWidth > 1) {
      o(llvm::formatv("v{0}", vecWidth), predicate);
    }
    return *this;
  }

  // Add ".b" suffix to instruction
  PtxIOInstr &b(int width) {
    o(llvm::formatv("b{0}", width));
    return *this;
  }

  PtxIOInstr &addAddr(mlir::Value addr, StringRef constraint, int off = 0) {
    auto *operand = newAddrOperand(addr, constraint, off);
    addOperand(operand);
    return *this;
  }

  Operand *newAddrOperand(mlir::Value addr, StringRef constraint, int off = 0);
};

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ASM_FORMAT_H_
