#ifndef TRITONINSTRUMENT_FUNCTIONBUILDER_H
#define TRITONINSTRUMENT_FUNCTIONBUILDER_H

#include "triton/Dialect/TritonInstrument/IR/Utility.h"

#include <string>
#include <variant>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class ImplicitLocOpBuilder;
class ModuleOp;
class Operation;
class RankedTensorType;
class Type;
class Value;
} // namespace mlir

namespace mlir::triton {
class FuncOp;

namespace instrument {

class ManglingArgs {
public:
  using Arg = std::variant<Type, int, std::string>;

  ManglingArgs() = default;
  ManglingArgs(const ManglingArgs &) = default;
  ManglingArgs(ManglingArgs &&) = default;
  ManglingArgs &operator=(const ManglingArgs &) = default;
  ManglingArgs &operator=(ManglingArgs &&) = default;

  ManglingArgs(std::initializer_list<Arg> args) : args(args) {}

  ~ManglingArgs() = default;

  template <typename T> void append(T arg) { args.push_back(arg); }

  template <typename T> void append(ArrayRef<T> arg) {
    for (auto &a : arg) {
      args.push_back(a);
    }
  }

  void append(ManglingArgs &other) {
    args.append(other.args.begin(), other.args.end());
  }

  std::string mangleArg(Arg arg) const {
    if (auto type = std::get_if<Type>(&arg)) {
      auto hash = static_cast<uint64_t>(mlir::hash_value(*type));
      return std::string("_T") + llvm::utohexstr(hash);
    } else if (auto intVal = std::get_if<int>(&arg)) {
      return std::string("_I") + std::to_string(*intVal);
    } else if (auto stringVal = std::get_if<std::string>(&arg)) {
      return *stringVal;
    }
    llvm_unreachable("Unsupported argument type");
  }

  std::string mangle(std::string baseName, int numWarps) const {
    std::string name = "__triton_consan_";
    name += baseName;
    name += "_nw" + std::to_string(numWarps);
    for (auto arg : args)
      name += mangleArg(arg);
    return name;
  }

private:
  SmallVector<Arg> args;
};

/// Utility to mangle helper function names produced by the instrumentation
/// passes. The mangled name encodes the base name, number of warps and the
/// participating types.
std::string mangleInstrumentHelperName(const std::string &baseName,
                                       int numWarps,
                                       llvm::ArrayRef<Type> types);

class FunctionBuilder {
public:
  FunctionBuilder(ModuleOp module, AuxDataMap &auxData)
      : module(module), auxData(auxData) {}

  // setWaiting: mark the base thread as waiting on the given barrier phase and
  // record that phase for deadlock detection.
  void createSetWaitingCall(ImplicitLocOpBuilder &b, Value mbar, int thread,
                            Value phase, Value pred, Operation *insertPoint);
  // clearWaiting: clear the waiting flag and stored phase for the base thread.
  void createClearWaitingCall(ImplicitLocOpBuilder &b, Value mbar, int thread,
                              Value pred, Operation *insertPoint);
  // checkAllActiveWaiting: assert that not all active threads are waiting on
  // matching barrier phases.
  void createCheckAllActiveWaitingCall(ImplicitLocOpBuilder &b, int activeMask,
                                       Value pred, Operation *insertPoint);
  // initBarrierState: Initialize the tracked barrier state to phase 0 and set
  // both the initial and current arrival counts.
  void createInitBarrierStateCall(ImplicitLocOpBuilder &b, Value mbar,
                                  int count, Operation *insertPoint);
  // verifyBarrierArrive: Check that applying the arrive count would not drive
  // the tracked current count negative. Triggers an assertion on failure.
  void createVerifyBarrierArriveCall(ImplicitLocOpBuilder &b, Value mbar,
                                     int count, Value pred,
                                     Operation *insertPoint);
  // updateBarrierState: Apply an arrive count to the tracked barrier state,
  // toggling the phase when the count reaches zero and reloading the current
  // count from the initial count.
  void createUpdateBarrierStateCall(ImplicitLocOpBuilder &b, Value mbar,
                                    int count, Value pred,
                                    Operation *insertPoint);
  // setWriteVisibility: Set the write visibility for a buffer. Marks the buffer
  // as visible to the threads set in threadMask. Clears out any other threads
  // from the visibility bitmask. We know this is safe because there cannot be
  // outstanding writes to this buffer at this point.
  void createSetWriteVisibilityCall(ImplicitLocOpBuilder &b, Value buf,
                                    uint32_t length, uint64_t threadMask,
                                    Value pred, MemType memType,
                                    Operation *insertPoint);
  // setReadVisibility: add the threads set in threadMask to the buffer's read
  // visibility bitmask.
  void createSetReadVisibilityCall(ImplicitLocOpBuilder &b, Value buf,
                                   uint32_t length, uint64_t threadMask,
                                   Value pred, MemType memType,
                                   Operation *insertPoint);
  // clearWriteTracking: clear all the information about threads writing to a
  // buffer.
  void createClearWriteTrackingCall(ImplicitLocOpBuilder &b, Value buf,
                                    uint32_t length, Value pred,
                                    MemType memType, Operation *insertPoint);
  // clearReadVisibility: clear the read visibility for a buffer.
  void createClearReadVisibilityCall(ImplicitLocOpBuilder &b, Value buf,
                                     uint32_t length, Value pred,
                                     MemType memType, Operation *insertPoint);
  // clearReadTracking: clear the read tracking for a buffer.
  void createClearReadTrackingCall(ImplicitLocOpBuilder &b, Value buf,
                                   uint32_t length, Value pred, MemType memType,
                                   Operation *insertPoint);
  // trackVisibleWrites: snapshot buffers currently visible to the thread into
  // the tracking table for a barrier.
  void createTrackVisibleWritesCall(ImplicitLocOpBuilder &b, Value mbar,
                                    int thread, Value pred, MemType memType,
                                    Operation *insertPoint);
  // trackVisibleReads: snapshot buffers currently visible to the thread into
  // the read tracking table for a barrier.
  void createTrackVisibleReadsCall(ImplicitLocOpBuilder &b, Value mbar,
                                   int thread, Value pred, MemType memType,
                                   Operation *insertPoint);
  // transferVisibleWrites: transfer write visibility tracked by a barrier to
  // all threads in threadMask.
  void createTransferVisibleWritesCall(ImplicitLocOpBuilder &b, Value mbar,
                                       uint64_t threadMask, Value pred,
                                       MemType memType, Operation *insertPoint);
  // transferVisibleReads: transfer read visibility tracked by a barrier to all
  // threads in threadMask.
  void createTransferVisibleReadsCall(ImplicitLocOpBuilder &b, Value mbar,
                                      uint64_t threadMask, Value pred,
                                      MemType memType, Operation *insertPoint);
  // verifyWriteVisibility: ensure the thread either sees the latest write or no
  // other thread is writing the buffer.
  void createVerifyWriteVisibilityCall(ImplicitLocOpBuilder &b, Value buf,
                                       uint32_t length, int thread,
                                       StringRef operandName, Value pred,
                                       MemType memType, Operation *insertPoint);
  // verifyReadVisibility: ensure all reads from the buffer are visible to the
  // thread.
  void createVerifyReadVisibilityCall(ImplicitLocOpBuilder &b, Value buf,
                                      uint32_t length, int thread,
                                      StringRef operandName, Value pred,
                                      MemType memType, Operation *insertPoint);
  // copyWriteVisibility: replicate the write visibility bit of sourceThread to
  // every destination thread in destMask.
  void createCopyWriteVisibilityCall(ImplicitLocOpBuilder &b, int sourceThread,
                                     uint64_t destMask, Value pred,
                                     MemType memType, Operation *insertPoint);
  // copyReadVisibility: replicate the read visibility row of sourceThread to
  // every destination thread in destMask.
  void createCopyReadVisibilityCall(ImplicitLocOpBuilder &b, int sourceThread,
                                    uint64_t destMask, Value pred,
                                    MemType memType, Operation *insertPoint);
  // stageAccessForCommit: mark the buffer as staged (value -1) in the
  // outstanding commit table for this thread.
  void createStageAccessForCommitCall(ImplicitLocOpBuilder &b, Value buf,
                                      uint32_t length, int thread, Value pred,
                                      MemType memType,
                                      CommitKind::Kind commitKind,
                                      Operation *insertPoint);
  // commitAccesses: convert staged entries to 1 and increment outstanding
  // commits greater than zero for the committing thread.
  void createCommitAccessesCall(ImplicitLocOpBuilder &b, int thread, Value pred,
                                CommitKind::Kind commitKind,
                                Operation *insertPoint);
  // clearOutstandingCommitsTransferWrites: clear entries farther than
  // outstandingNum from the thread and set write visibility for threads in
  // transferThreadMask.
  void createClearOutstandingCommitsTransferWritesCall(
      ImplicitLocOpBuilder &b, int thread, uint64_t transferThreadMask,
      int outstandingNum, Value pred, CommitKind::Kind commitKind,
      MemType memType, Operation *insertPoint);
  // clearOutstandingCommitsTransferReads: clear entries farther than
  // outstandingNum from the thread and set read visibility for threads in
  // transferThreadMask.
  void createClearOutstandingCommitsTransferReadsCall(
      ImplicitLocOpBuilder &b, int thread, uint64_t transferThreadMask,
      int outstandingNum, Value pred, CommitKind::Kind commitKind,
      MemType memType, Operation *insertPoint);
  // checkOutstandingCommits: assert that the outstanding commit row for the
  // buffer is zero before the access described by pendingAccessType.
  void createCheckOutstandingCommitsCall(ImplicitLocOpBuilder &b, Value buf,
                                         uint32_t length, int thread,
                                         StringRef pendingAccessType,
                                         Value pred, MemType memType,
                                         CommitKind::Kind commitKind,
                                         Operation *insertPoint);

private:
  ModuleOp module;
  AuxDataMap &auxData;
};

} // namespace instrument
} // namespace mlir::triton

#endif
