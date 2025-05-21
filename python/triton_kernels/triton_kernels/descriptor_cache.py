import torch
from triton.tools.tensor_descriptor import TensorDescriptor
from typing import Dict, Any, Tuple, Optional


class DescriptorCache:
    """Cache of tensor descriptors"""

    def __init__(self):
        self.cache: Dict[Any, TensorDescriptor] = {}

    def get_or_create(self, key: Any, create_fn):
        """Get descriptor from cache or create it"""
        if key not in self.cache:
            self.cache[key] = create_fn()
        return self.cache[key]

    def clear(self):
        """Clear all cached descriptors"""
        self.cache.clear()


class TensorCache:
    """Cache of tensors"""

    def __init__(self):
        self.cache: Dict[str, torch.Tensor] = {}

    def get_or_create(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Get tensor from cache or create it"""
        key = f"{shape}_{dtype}_{device}"
        if key not in self.cache:
            self.cache[key] = torch.empty(shape, dtype=dtype, device=device)
        return self.cache[key]

    def clear(self):
        """Clear all cached tensors"""
        self.cache.clear()


class TensorDescriptorBuilder:
    """Builder for creating different types of tensor descriptors"""

    @staticmethod
    def create_basic_descriptor(tensor: torch.Tensor, block_shape: Tuple[int, ...],
                                transpose: bool = False) -> TensorDescriptor:
        """Create a basic tensor descriptor with optional transpose"""
        if transpose:
            block_shape = block_shape[:-2] + [block_shape[-1], block_shape[-2]]
            tensor = tensor.permute(0, 2, 1)
        return TensorDescriptor.from_tensor(tensor, block_shape=block_shape)

    @staticmethod
    def create_weight_descriptor(w_tensor: torch.Tensor, block_k: int, block_n: int,
                                 transpose: bool) -> TensorDescriptor:
        """Create a tensor descriptor for weight matrix"""
        # Two e2m1 packed in a uint8 or a single fp8
        W_PACK_DIVISOR = 2 if w_tensor.dtype == torch.uint8 else 1
        PACKED_BLOCK_K_W = block_k // W_PACK_DIVISOR
        return TensorDescriptorBuilder.create_basic_descriptor(w_tensor, block_shape=[1, PACKED_BLOCK_K_W, block_n],
                                                               transpose=transpose)

    @staticmethod
    def create_block_scale_descriptor(mx_tensor: torch.Tensor, block_k: int, block_n: int, K: int, N: int,
                                      mx_scale_stride_k: int, mx_scale_stride_n: int, n_expts_tot: int, batch_size: int,
                                      expt_data_blocks: Optional[Any], swizzle_mx: bool,
                                      transpose: bool) -> TensorDescriptor:
        """Create a tensor descriptor for block scale factors"""
        MX_PACK_DIVISOR = 32
        MX_SCALE_BLOCK_K = block_k // MX_PACK_DIVISOR
        PackedK = (K + MX_PACK_DIVISOR - 1) // MX_PACK_DIVISOR

        if swizzle_mx:
            num_expt_x_ncol = (n_expts_tot if len(expt_data_blocks) > 0 else batch_size) * ((N + 127) // 128)
            return TensorDescriptor(
                base=mx_tensor, shape=[1, num_expt_x_ncol, (PackedK + 3) // 4, 2, 256],
                strides=[num_expt_x_ncol * mx_scale_stride_n, mx_scale_stride_n, mx_scale_stride_k, 256,
                         1], block_shape=[1, block_n // 128, MX_SCALE_BLOCK_K // 4, 2, 256])
        else:
            # Non-optimal SF layout, expect slow transfers
            # from global to shmem and from shmem to tmem
            return TensorDescriptorBuilder.create_basic_descriptor(mx_tensor,
                                                                   block_shape=[1, MX_SCALE_BLOCK_K,
                                                                                block_n], transpose=transpose)

    @staticmethod
    def create_input_descriptor_gather(x_tensor: torch.Tensor, K: int, x_stride_1: int, x_stride_2: int,
                                       block_k: int) -> TensorDescriptor:
        """Create a tensor descriptor for input matrix X via TMA gather"""
        x_desc = x_tensor.squeeze()
        assert x_desc.ndim == 2, "TMA gather descriptor requires 2D input"
        INT_MAX = 2147483647
        return TensorDescriptor(base=x_desc, shape=[INT_MAX, K], strides=[x_stride_1, x_stride_2],
                                block_shape=[1, block_k])

    @staticmethod
    def create_input_descriptor_load(x_tensor: torch.Tensor, K: int, x_stride_1: int, x_stride_2: int, block_m: int,
                                     block_k: int) -> TensorDescriptor:
        """Create a tensor descriptor for input matrix X via TMA"""
        x_desc = x_tensor.squeeze()
        assert x_desc.ndim == 2, "LHS input TMA descriptor builder expects 2D input"
        return TensorDescriptor(base=x_desc, shape=[x_desc.shape[0], K], strides=[x_stride_1, x_stride_2],
                                block_shape=[block_m, block_k])

    @staticmethod
    def create_input_descriptor(x_tensor: torch.Tensor, K: int, x_stride_1: int, x_stride_2: int, block_k: int,
                                block_m: int, use_gather_tma: bool, use_load_tma: bool) -> TensorDescriptor:
        """Create a tensor descriptor for input matrix X based on TMA usage"""
        if use_gather_tma:
            return TensorDescriptorBuilder.create_input_descriptor_gather(x_tensor, K, x_stride_1, x_stride_2, block_k)
        elif use_load_tma:
            return TensorDescriptorBuilder.create_input_descriptor_load(x_tensor, K, x_stride_1, x_stride_2, block_m,
                                                                        block_k)
        else:
            raise ValueError("Host TMA descriptors requires LHS to use TMA")


class CacheManager:

    def __init__(self):
        self.descriptor_cache = DescriptorCache()
        self.tensor_cache = TensorCache()
        self.builder = TensorDescriptorBuilder()

    def clear_all(self):
        """Clear all caches"""
        self.descriptor_cache.clear()
        self.tensor_cache.clear()
