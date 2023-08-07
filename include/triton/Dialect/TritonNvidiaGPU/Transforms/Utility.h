/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
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

#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {

// 0 is reserved for default sync.
// TODO: comprehensive mechanism to globally manage namedbarrier.
static int const nameBarrierIdBegin = 1;
static int nameBarrierIdEnd = 16;

/// Helper functions for async agent
typedef int AgentId;
SmallVector<AgentId> getAgentIds(Operation *op);
bool hasAgentId(Operation *op, AgentId agentId);
void setAgentIds(Operation *op, ArrayRef<AgentId> agentIds);
SmallVector<AgentId> collectAgentIds(Operation *op);
void addAgentIds(Operation *op, ArrayRef<int> agents);
SmallVector<int> getMutexBarIds(Operation *op);
SmallVector<int> getMutexNumThreads(Operation *op);

class OpBuilderWithAgentIds : public OpBuilder {
public:
  OpBuilderWithAgentIds(MLIRContext *context) : OpBuilder(context) {}

  void setAgentIdsFromArray(ArrayRef<AgentId> newAgentIds) {
    agentIds = SmallVector<AgentId>(newAgentIds.begin(), newAgentIds.end());
  }

  void setAgentIdsFromOp(Operation *op) {
    setAgentIdsFromArray(getAgentIds(op));
  }

  void setAgentIdsFromValueUsers(Value value) {
    SetVector<AgentId> agentIdSet;
    for (Operation *user : value.getUsers())
      for (AgentId agentId : getAgentIds(user))
        agentIdSet.insert(agentId);
    setAgentIdsFromArray(agentIdSet.getArrayRef());
  }

  template <typename OpTy, typename... Args>
  OpTy createWithAgentIds(Args &&...args) {
    OpTy op = create<OpTy>(std::forward<Args>(args)...);
    if (!agentIds.empty())
      setAgentIds(op, agentIds);
    return op;
  }

private:
  SmallVector<AgentId> agentIds;
};

/// Constant agent ids
constexpr AgentId kLoadAgentId = 0;
constexpr AgentId kDotAgentId = 1;

bool isWSCandidateLoad(Operation *op);
bool isWSSupported(ModuleOp m, int computeCapability);

LogicalResult getDependentValues(Value val, DenseSet<Value> &depSet,
                                 const DenseSet<Value> &stopSet = {});
LogicalResult getDependentValues(Operation *op, DenseSet<Value> &depSet,
                                 const DenseSet<Value> &stopSet = {});
DenseSet<Operation *> getDependentOps(DenseSet<Value> &depSet);

} // namespace mlir

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_UTILITY_H_
