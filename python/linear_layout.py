import numpy as np
import math
from dataclasses import dataclass
from typing import List


@dataclass
class blockedEncodingAttr:
    sizePerThread: List[int]
    threadsPerWarp: List[int]
    warpsPerCTA: List[int]
    order: List[int]


@dataclass
class SharedEncodingAttr:
    vec: int
    perPhase: int
    maxPhase: int
    order: List[int]


@dataclass
class AMDMfmaEncodingAttr:
    warpsPerCTA: List[int]
    instrShape: List[int]
    isTranspose: bool


def base_to_linear_layout(
        base: dict[int, list[int]]) -> tuple[np.array, np.array]:
    rows = np.array([val[0] for _, val in base.items()])
    cols = np.array([val[1] for _, val in base.items()])
    log_rows = np.log2(rows) + 1
    log_cols = np.log2(cols) + 1
    log_rows[np.isinf(log_rows)] = 0
    log_cols[np.isinf(log_cols)] = 0

    num_ll_cols = len(base)
    num_ll_rows = int(log_rows.max() + log_cols.max())

    linear_layout_decimal = np.array(list(base.values())).T
    linear_layout_binary = np.zeros((num_ll_rows, num_ll_cols), dtype=int)
    for i, (r, c) in enumerate(base.values()):
        # print("r:", r, "c:" , c)
        if r != 0:
            idx = int(math.log2(r))
            linear_layout_binary[idx, i] = 1
        if c != 0:
            idx = int(math.log2(c)) + int(log_rows.max())
            linear_layout_binary[idx, i] = 1

    return linear_layout_binary, linear_layout_decimal


def ll_map(hw_idx, ll, order='minor_first'):
    num_cols = ll.shape[-1]
    assert hw_idx <= 2**num_cols - 1, \
        f"index {hw_idx} exceeds the shared layout can support"
    # convert hardware index from decimal to binary
    idx_binary = np.zeros((num_cols, 1), dtype=int)
    idxb = np.binary_repr(hw_idx)
    for j, binary in enumerate(idxb[::-1]):
        idx_binary[j] = binary

    return (ll @ idx_binary).squeeze()


def blocked_to_ll(hw_idx: list[int], ll_reg, ll_lane, ll_warp):
    num_cols_reg = ll_reg.shape[-1]
    num_cols_lane = ll_lane.shape[-1]
    num_cols_warp = ll_warp.shape[-1]
    cols_offset = np.cumsum(
        np.array([0, num_cols_reg, num_cols_lane, num_cols_warp]))

    assert hw_idx[0] <= 2**num_cols_reg - 1, \
        f"register index {hw_idx} exceeds the register layout can support"
    assert hw_idx[1] <= 2**num_cols_lane - 1, \
        f"register index {hw_idx} exceeds the lane layout can support"
    assert hw_idx[2] <= 2**num_cols_warp - 1, \
        f"register index {hw_idx} exceeds the warp layout can support"

    ll_full = np.concatenate((ll_reg, ll_lane, ll_warp), axis=1)
    num_cols = num_cols_reg + num_cols_lane + num_cols_warp
    idx_binary = np.zeros((num_cols, 1), dtype=int)
    for i, idx in enumerate(hw_idx):
        idxb = np.binary_repr(idx)
        for j, binary in enumerate(idxb[::-1]):
            idx_binary[cols_offset[i] + j] = binary
    # print(idx_binary)
    return (ll_full @ idx_binary).squeeze()


# def blocked_to_ll(hw_idx: list[int], ll_reg, ll_lane):
#     num_cols_reg = ll_reg.shape[-1]
#     num_cols_lane = ll_lane.shape[-1]
#     cols_offset = np.cumsum(np.array([0, num_cols_reg, num_cols_lane]))

#     assert hw_idx[0] <= 2**num_cols_reg - 1, \
#         f"register index {hw_idx} exceeds the register layout can support"
#     assert hw_idx[1] <= 2**num_cols_lane - 1, \
#         f"register index {hw_idx} exceeds the lane layout can support"

#     ll_full = np.concatenate((ll_reg, ll_lane), axis=1)
#     idx_binary = np.zeros((num_cols_reg+num_cols_lane, 1), dtype=int)
#     for i, idx in enumerate(hw_idx):
#         idxb = np.binary_repr(idx)
#         for j, binary in enumerate(idxb[::-1]):
#             idx_binary[cols_offset[i] + j] = binary
#     # print(idx_binary)
#     return (ll_full @ idx_binary).squeeze()


reg_layout = {
    1: [0, 1],
    2: [0, 2],
    4: [1, 0],
    8: [2, 0],
    16: [4, 0],
    # 32: [32, 0],
}

lane_layout = {
    1: [0, 8],
    2: [0, 16],
    4: [0, 32],
    8: [4, 0],
    16: [8, 0],
    32: [16, 0],
}

# warp_layout = {
#     1:  [0, 16],
#     2:  [0, 32],
# }

# llb_reg, lld_reg = base_to_linear_layout(reg_layout)
# # print("reg layout")
# print(lld_reg)
# print(llb_reg)

# llb_lane, lld_lane = base_to_linear_layout(lane_layout)
# print("lane layout")
# print(lld_lane)

# llb_warp, lld_warp = base_to_linear_layout(warp_layout)
# print("warp layout")
# print(lld_warp)

# print("blocked layout")
# print(np.concatenate((lld_reg, lld_lane, lld_warp), axis=1))

# ll_dict = {}
# sw_row, sw_col = 0, 0
# for lane in range(1):
#     for reg in range(64):
#         sw_idx = tuple(blocked_to_ll([reg, lane], lld_reg, lld_lane).tolist())
#         assert sw_idx not in ll_dict.keys()
#         # leading number from sw_idx is the fastest changing dim, which is usually column
#         sw_row = max(sw_row, int(sw_idx[1]))
#         sw_col = max(sw_col, int(sw_idx[0]))
#         ll_dict[sw_idx] = (reg, lane)
#         print(f"({reg:2d}, {lane}) -> {sw_idx}")

# print("shape from the ll:", sw_row+1, sw_col+1)

# # for r in range(sw_row+1):
# #     for c in range(sw_col+1):
# #         if (c, r) not in ll_dict.keys():
# #             continue
# #         reg, lane, warp = ll_dict[(c, r)]
# #         print(f"t{lane:<2d}r{reg%8:<2d}", end=" ")
# #         # print(f"t{lane:<2d}r{reg:<2d}w{warp}", end=" ")
# #     print()


""" SharedLayout """
numBanks = 32
bankBitWidth = 32
SIMDWidth = 16

shared_layout = {
    1: [1, 0],
    2: [2, 0],
    4: [4, 0],
    8: [8, 0],
    16: [16, 0],
    32: [32, 0],
    64: [4, 1],
    128: [8, 2],
    256: [16, 4],
    512: [32, 8],
    1024: [0, 16],
    2048: [0, 32],
    4096: [0, 64],
    8192: [0, 128],
}

llb_shared, lld_shared = base_to_linear_layout(shared_layout)
print("shared layout")
print(lld_shared)

typeWidthInBit = 16
elemsPerOneBanksRow = numBanks * bankBitWidth / typeWidthInBit
# for offset in range(256):
#     # print(f"{offset} -> {ll_map(offset, lld_shared)}", end=", ")
#     print(f"{ll_map(offset, lld_shared)[0]}", end=", ")
#     if (offset + 1) % elemsPerOneBanksRow == 0:
#         print()

print()
print(f"offset={68} -> {ll_map(68, lld_shared)}")
print(f"offset={64} -> {ll_map(64, lld_shared)}")

vec = 4
perPhase = 1
maxPhase = 8
K = 64
N = 4
swizzle = True
for n in range(N):
    phase = (n // perPhase) % maxPhase
    for k in range(K):
        colSwizzled = ((k // vec) ^ phase) * vec + k % vec
        idx = colSwizzled if swizzle else k
        # print(f"({n:>3d}, {idx:>2d})", end=" ")
        index = n * K+idx
        print(f"{index:>3d}", end=" ")
    print()

# for n in range(N):
#     phase = (n // perPhase) % maxPhase
#     for k in range(K):
#         colSwizzled = ((k // vec) ^ phase) * vec + k % vec
#         idx = colSwizzled if swizzle else k
#         index = n*K+idx
#         # if index == 68:
#         #     print(index, k, n)
#         if index & (index-1) == 0 and index:
#             print(f"offset={index} -> ({k}, {n})")
