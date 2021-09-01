import numpy as np


class PhiloxConfig:
    def __init__(self, PHILOX_ROUND_A, PHILOX_ROUND_B, PHILOX_KEY_A, PHILOX_KEY_B, DTYPE):
        self.PHILOX_ROUND_A = np.array(PHILOX_ROUND_A, dtype=DTYPE)
        self.PHILOX_ROUND_B = np.array(PHILOX_ROUND_B, dtype=DTYPE)
        self.PHILOX_KEY_A = np.array(PHILOX_KEY_A, dtype=DTYPE)
        self.PHILOX_KEY_B = np.array(PHILOX_KEY_B, dtype=DTYPE)
        self.DTYPE = DTYPE


# This is better for GPU
PHILOX_32 = PhiloxConfig(
    PHILOX_KEY_A=0x9E3779B9,
    PHILOX_KEY_B=0xBB67AE85,
    PHILOX_ROUND_A=0xD2511F53,
    PHILOX_ROUND_B=0xCD9E8D57,
    DTYPE=np.uint32,
)

# This is what numpy implements
PHILOX_64 = PhiloxConfig(
    PHILOX_KEY_A=0x9E3779B97F4A7C15,
    PHILOX_KEY_B=0xBB67AE8584CAA73B,
    PHILOX_ROUND_A=0xD2E7470EE14C6C93,
    PHILOX_ROUND_B=0xCA5A826395121157,
    DTYPE=np.uint64,
)


class CustomPhilox4x:
    def __init__(self, seed, config):
        self._config = config
        seed = self._into_pieces(seed)
        self._key = np.array(seed[:2], dtype=self._dtype)
        self._counter = np.array((0, 0) + seed[2:], dtype=self._dtype)

    @property
    def _dtype(self):
        return self._config.DTYPE

    def _into_pieces(self, n, pad=4):
        res = []
        while len(res) < pad:
            res.append(np.array(n, dtype=self._dtype))
            n >>= (np.dtype(self._dtype).itemsize * 8)
        assert n == 0
        return tuple(res)

    def _multiply_low_high(self, a, b):
        low = a * b
        high = int(a) * int(b)
        high = np.array(high >> (np.dtype(self._dtype).itemsize * 8), dtype=self._dtype)
        return low, high

    def _single_round(self, counter, key):
        lo0, hi0 = self._multiply_low_high(self._config.PHILOX_ROUND_A, counter[0])
        lo1, hi1 = self._multiply_low_high(self._config.PHILOX_ROUND_B, counter[2])

        return np.array(
            [
                hi1 ^ counter[1] ^ key[0],
                lo1,
                hi0 ^ counter[3] ^ key[1],
                lo0,
            ],
            dtype=self._dtype
        )

    def _raise_key(self, key):
        return np.array(
            [
                key[0] + self._config.PHILOX_KEY_A,
                key[1] + self._config.PHILOX_KEY_B,
            ],
            dtype=self._dtype
        )

    def random_raw(self):
        counter = self._counter
        key = self._key

        for _ in range(10):
            counter = self._single_round(counter, key)
            key = self._raise_key(key)

        self.advance(1)
        return counter

    def advance(self, n_steps):
        self._counter[0] += n_steps
        assert self._counter[0] < 2**32, "FIXME: doesn't work for large offsets"


class CustomPhilox(CustomPhilox4x):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = []

    def random_raw(self):
        if len(self.buffer) == 0:
            self.buffer = list(super().random_raw())[::-1]
        return int(self.buffer.pop())
