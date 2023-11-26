# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

from ..runtime import driver


def generate_cu_signature(constants, signature, ids):
    # CUtensorMap*s are always the last arguments
    num_regular_signatures = max(signature.keys()) + 1 if len(signature) > 0 else 0
    if ids["ids_of_tensormaps"] is not None:
        for i, _ in enumerate(ids["ids_of_tensormaps"]):
            signature[num_regular_signatures + i] = '*CUtensorMap'
    return signature, num_regular_signatures


def dummy_tensormaps_info(n=2):
    ret = []
    for i in range(n):
        ret.append(InfoFromBackendForTensorMap(dummy=True))
    return ret


def parse_tma_info(infos, ids_of_folded_args):
    ret = []
    for info in infos:
        e = InfoFromBackendForTensorMap(infos=info)
        e.ids_of_folded_args = ids_of_folded_args
        ret.append(e)
    return ret


def get_tma_mapping(tensormaps_info):
    ret = {}
    if tensormaps_info is not None:
        for i, e in enumerate(tensormaps_info):
            ret.update(e.get_address_tma_mapping())
    else:
        ret = None
    return ret


def get_ids_of_tensormaps(tensormaps_info):
    ret = None
    # order is not relevant
    if tensormaps_info is not None:
        ret = [e.get_id_of_tensormap() for e in tensormaps_info]
    return ret


# decouple information for tensormap from backend
# please ignore the naming style, xx_yy is compiler.py style, xxYy is to comply with cuda tensormap style
# mixing style is for readability
class InfoFromBackendForTensorMap:
    N = 2
    n = 0
    ntma = 0

    def __init__(self, infos=None, dummy=False):
        self.dummy = dummy
        self.ids_of_folded_args = ()
        if not dummy and not isinstance(infos, dict):
            self._extract_info_from_backend(infos)
        elif not dummy and isinstance(infos, dict):
            self._extract_info_from_dict(infos)
        elif dummy:
            self._dummy()

    def _dummy(self):
        assert InfoFromBackendForTensorMap.n < InfoFromBackendForTensorMap.N
        if InfoFromBackendForTensorMap.n == 0:
            self.tensorDataType = driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_FLOAT16"]
            self.tensorRank = 4
            self.globalAddressArgIdx = 0
            self.globalStridesArgIdx = [7, 6, -1, -1]
            self.globalDimsArgIdx = [5, 3, -1, -1]
            self.boxDims = [16, 64, 1, 1]
            self.elementStrides = [1, 1, 1, 1]
            self.interleave = driver.utils.CUtensorMapInterleave["CU_TENSOR_MAP_INTERLEAVE_NONE"]
            self.swizzle = driver.utils.CUtensorMapSwizzle["CU_TENSOR_MAP_SWIZZLE_32B"]
            self.l2Promotion = driver.utils.CUtensorMapL2promotion["CU_TENSOR_MAP_L2_PROMOTION_L2_128B"]
            self.TMADescArgIdx = 11
            self.oobFill = driver.utils.CUtensorMapFloatOOBfill["CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE"]
            InfoFromBackendForTensorMap.n += 1
            return
        if InfoFromBackendForTensorMap.n == 1:
            self.tensorDataType = driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_FLOAT16"]
            self.tensorRank = 4
            self.globalAddressArgIdx = 1
            self.globalStridesArgIdx = [7, 6, -1, -1]
            self.globalDimsArgIdx = [5, 3, -1, -1]
            self.boxDims = [16, 64, 1, 1]
            self.elementStrides = [1, 1, 1, 1]
            self.interleave = driver.utils.CUtensorMapInterleave["CU_TENSOR_MAP_INTERLEAVE_NONE"]
            self.swizzle = driver.utils.CUtensorMapSwizzle["CU_TENSOR_MAP_SWIZZLE_32B"]
            self.l2Promotion = driver.utils.CUtensorMapL2promotion["CU_TENSOR_MAP_L2_PROMOTION_L2_128B"]
            self.TMADescArgIdx = 12
            self.oobFill = driver.utils.CUtensorMapFloatOOBfill["CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE"]
            InfoFromBackendForTensorMap.n += 1
            return

    def _extract_info_from_backend(self, infos):
        self.tensorDataType = infos.tensorDataType
        self.tensorRank = infos.tensorRank
        self.globalAddressArgIdx = infos.globalAddressArgIdx
        self.globalStridesArgIdx = infos.globalStridesArgIdx
        self.globalDimsArgIdx = infos.globalDimsArgIdx
        self.boxDims = infos.boxDims
        self.elementStrides = infos.elementStrides
        self.interleave = infos.interleave
        self.swizzle = infos.swizzle
        self.l2Promotion = infos.l2Promotion
        self.oobFill = infos.oobFill
        self.TMADescArgIdx = infos.TMADescArgIdx

    # dict could be from cached metadata json
    def _extract_info_from_dict(self, infos: dict):
        self.tensorDataType = infos['tensorDataType']
        self.tensorRank = infos['tensorRank']
        self.globalAddressArgIdx = infos['globalAddressArgIdx']
        self.globalStridesArgIdx = infos['globalStridesArgIdx']
        self.globalDimsArgIdx = infos['globalDimsArgIdx']
        self.boxDims = infos['boxDims']
        self.elementStrides = infos['elementStrides']
        self.interleave = infos['interleave']
        self.swizzle = infos['swizzle']
        self.l2Promotion = infos['l2Promotion']
        self.oobFill = infos['oobFill']
        self.TMADescArgIdx = infos['TMADescArgIdx']

    def get_address_tma_mapping(self):
        return {self.globalAddressArgIdx: self.TMADescArgIdx + len(self.ids_of_folded_args)}

    def get_id_of_tensormap(self):
        return self.TMADescArgIdx + len(self.ids_of_folded_args)

    def getTMADescArgIdx(self):
        return self.TMADescArgIdx

    # dtype:cuda.CUtensorMapDataType | int
    def bytes_from_type(self, dtype):
        return {
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_UINT8"]: 1,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_UINT16"]: 2,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_UINT32"]: 4,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_INT32"]: 4,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_UINT64"]: 8,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_INT64"]: 8,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_FLOAT16"]: 2,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_FLOAT32"]: 4,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_FLOAT64"]: 8,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_BFLOAT16"]: 2,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ"]: 4,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_TFLOAT32"]: 4,
            driver.utils.CUtensorMapDataType["CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ"]: 4
        }[dtype]

    def getTensorMapDataType(self):
        return self.tensorDataType

    def getInterleave(self):
        return self.interleave

    def getSwizzle(self):
        return self.swizzle

    def getL2Promotion(self):
        return self.l2Promotion

    def getOobFill(self):
        return self.oobFill

    def getTensorRank(self):
        return self.tensorRank

    def getBoxDims(self):
        return self.boxDims

    def getElementStrides(self):
        return self.elementStrides

    def getGlobalAddress(self, args):
        idx = self.getOriginArgIdx(self.globalAddressArgIdx, args)
        return args[idx]

    # args, captured kernel args in runtime
    def getGlobalDims(self, args):
        shape = []
        for e in self.globalDimsArgIdx:
            t = 1
            # < 0 means folded arg or constant (-1 - value)
            # -1 means extended dim which is 1, -2 means folded arg with constant 1 (-1 - value)
            if e == -1:
                t = 1
            elif e < 0 and e != -1:
                t = -e - 1
            else:
                idx = self.getOriginArgIdx(e, args)
                t = args[idx]
            shape.append(t)
        return shape

    def getGlobalStrides(self, args):
        t_globalDims = [int(e) for e in self.getGlobalDims(args)]
        t_globalStridesArgIdx = self.globalStridesArgIdx.copy()
        strides_in_elements = []
        # todo: get all stride from backend even in extended mode
        for i in range(self.tensorRank):
            t = 1
            if t_globalStridesArgIdx[i] == -1:
                for ii in range(i):
                    t *= t_globalDims[ii]
            # -2 means the sride in arguments is folded constant 1, we don't use 1 because it can not be distinguished from index 1
            elif t_globalStridesArgIdx[i] < 0:
                t = -1 - t_globalStridesArgIdx[i]
            else:
                new_idx = self.getOriginArgIdx(t_globalStridesArgIdx[i], args)
                t = args[new_idx]

            strides_in_elements.append(t)

        strides_in_elements = strides_in_elements[1:]
        strides_in_bytes = [e * self.bytes_from_type(self.tensorDataType) for e in strides_in_elements]
        return strides_in_bytes

    def getOriginArgIdx(self, idx, args):
        if self.ids_of_folded_args:
            ids_before_folding_arg = [i for i in range(len(args)) if i not in self.ids_of_folded_args]
            return ids_before_folding_arg[idx]
        else:
            return idx

    def tensormap(self, args):
        return driver.utils.cuTensorMapEncodeTiled(
            self.getTensorMapDataType(),
            self.getTensorRank(),
            self.getGlobalAddress(args),
            self.getGlobalDims(args),
            self.getGlobalStrides(args),
            self.getBoxDims(),
            self.getElementStrides(),
            self.getInterleave(),
            self.getSwizzle(),
            self.getL2Promotion(),
            self.getOobFill(),
        )

    # make hashable to use as partial key in cache
    def __hash__(self):
        return hash((self.ids_of_folded_args, self.globalAddressArgIdx, tuple(self.globalDimsArgIdx),
                     tuple(self.globalStridesArgIdx), self.tensorDataType, self.tensorRank, tuple(self.boxDims),
                     tuple(self.elementStrides), self.interleave, self.swizzle, self.l2Promotion, self.oobFill))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.ids_of_folded_args, self.globalAddressArgIdx, self.globalDimsArgIdx, self.globalStridesArgIdx,
                self.tensorDataType, self.tensorRank, self.boxDims, self.elementStrides, self.interleave, self.swizzle,
                self.l2Promotion,
                self.oobFill) == (other.ids_of_folded_args, other.globalAddressArgIdx, other.globalDimsArgIdx,
                                  other.globalStridesArgIdx, other.tensorDataType, other.tensorRank, other.boxDims,
                                  other.elementStrides, other.interleave, other.swizzle, other.l2Promotion,
                                  other.oobFill)


class TensorMapManager:

    def __init__(self):
        self.tensormaps_device = {}

    def __getitem__(self, key: tuple):
        if key in self.tensormaps_device:
            return int(self.tensormaps_device[key])
        else:
            (e, args) = key
            t_tensormap = e.tensormap(args)
            TENSORMAP_SIZE_IN_BYTES = 128
            t_tensormap_device = driver.utils.cuMemAlloc(TENSORMAP_SIZE_IN_BYTES)
            driver.utils.cuMemcpyHtoD(t_tensormap_device, t_tensormap, TENSORMAP_SIZE_IN_BYTES)
            self.tensormaps_device[key] = t_tensormap_device
            return int(self.tensormaps_device[key])

    def __del__(self):
        for _, v in self.tensormaps_device.items():
            driver.utils.cuMemFree(v)
