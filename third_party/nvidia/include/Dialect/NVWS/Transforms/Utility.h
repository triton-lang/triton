/*
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef NVWS_TRANSFORM_UTILITY_H_
#define NVWS_TRANSFORM_UTILITY_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace nvws {

constexpr char ATTR_WS_MANUAL[] = "nvws.manual";

constexpr char ATTR_WS_GROUPS[] = "groups";
constexpr char ATTR_WS_PREFIX[] = "nvws.group.";
constexpr char ATTR_WS_TMALOAD[] = "nvws.group.tma_load";
constexpr char ATTR_WS_MMA[] = "nvws.group.mma";
constexpr char ATTR_WS_EPILOGUE[] = "nvws.group.epilogue";

constexpr char ATTR_WS_AREF_IF[] = "aref_if";
constexpr char ATTR_WS_INIT_BARRIER_SYNC[] = "init_barrier_sync";
constexpr char ATTR_WS_BARID[] = "barId";

constexpr char ATTR_WS_START_WARP[] = "start_warp";
constexpr char ATTR_WS_NUM_WARPS[] = "num_warps";
constexpr char ATTR_WS_REG_COUNT[] = "reg_count";

bool isManuallyGrouped(Operation *op);
bool isManuallyGrouped(ModuleOp module);

class WSGroup {
public:
  WSGroup() = default;
  WSGroup(int startWarp, int numWarps, int regCount = 0)
      : startWarp(startWarp), numWarps(numWarps), regCount(regCount) {}

  int getStartWarp() const { return startWarp; }
  int getNumWarps() const { return numWarps; }
  bool hasRegCount() const { return regCount > 0; }
  int getRegCount() const { return regCount; }

private:
  int startWarp = 0;
  int numWarps = 0;
  int regCount = 0;
};

std::set<std::string> getGroups(Operation *op);
std::set<std::string> getGroups(OpResult result);
std::set<std::string> getGroupsIdx(mlir::Operation *op, int idx);

void setGroups(Operation *op, const std::set<std::string> &groups);
void setGroups(OpResult result, const std::set<std::string> &groups);
void setGroupsIdx(Operation *op, int idx, const std::set<std::string> &groups);

void addGroups(Operation *op, const std::set<std::string> &groups);
void addGroups(OpResult result, const std::set<std::string> &groups);
void addGroupsIdx(Operation *op, int idx, const std::set<std::string> &groups);

void copyGroups(Operation *from_op, Operation *to_op);

class OpBuilderWithGroup : public OpBuilder {
public:
  explicit OpBuilderWithGroup(Operation *op, std::string wsGroup = {})
      : OpBuilder(op), group(wsGroup) {}

  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args &&...args) {
    OpTy op = OpBuilder::create<OpTy>(location, std::forward<Args>(args)...);
    if (!group.empty()) {
      setGroups(op, {group});
    }
    return op;
  }
  using OpBuilder::create;

private:
  std::string group;
};

SmallVector<nvidia_gpu::WarpGroupOp> findWarpGroupOps(ModuleOp m);

void setGroupAttribute(ModuleOp moduleOp, const std::string &name,
                       WSGroup group);
SymbolRefAttr mkGroup(ModuleOp moduleOp, const std::string &name,
                      WSGroup group);

std::string getGroup(nvidia_gpu::WarpGroupOp wgOp);
WSGroup getGroupFromAttribute(Attribute attr);
WSGroup getGroupFromSymbolRefAttr(ModuleOp mod, SymbolRefAttr refAttr);

bool isOpInGroup(Operation *op, const std::string &group);
bool isResultInGroup(Value value, const std::string &group);
Value getLoopNumIter(Value lb, Value ub, Value step, Location loc,
                     OpBuilder &builder);
Value getLoopNumIter(scf::ForOp forOp, OpBuilder &builder);
int getBarrierID(nvidia_gpu::WarpGroupOp wgOp);

bool isMMAOp(Operation *op);
bool isInnerMostLoop(scf::ForOp forOp);
bool isFMHAMathLoop(scf::ForOp forOp);
std::map<std::string, WSGroup> collectGroups(ModuleOp mod);

struct TokenInfo {
  Operation *producerOp;
  Value buffer;
};
TokenInfo getTokenProducerOp(Value result);

gpu::MemDescType getDataMemDescType(gpu::MemDescType memDescType,
                                    bool mutableMemory);
gpu::MemDescType getArefbufMemDescType(gpu::MemDescType memDescType,
                                       int32_t AREF_SIZE);
bool isHopper(ModuleOp mod);
Value mkConstant(OpBuilder &builder, Location loc, int value, int width,
                 std::set<std::string> groups);
bool isConstant(Value value, int constant);

Operation *createAlloc(OpBuilder &builder, Location loc,
                       gpu::MemDescType memDescType, Value src);

bool isMMAOperandLoadOp(Operation *op);

} // namespace nvws
} // namespace triton
} // namespace mlir

#endif // NVWS_TRANSFORM_UTILITY_H_
