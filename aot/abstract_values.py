from dataclasses import dataclass

class AbstractValue:
    @property
    def tt_dtype(self):
        return self._tt_dtype

    def __eq__(self, o: object) -> bool:
        return self.num == o

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__.replace("Input", "")

        repr = f"<{self.tt_dtype}"
        if "Ptr" in cls_name:
            repr += "*"
        if hasattr(self, "num"):
            repr += f",{self.num}"
        repr += ">"
        return repr


class AbstractInt(AbstractValue, int):
    def __new__(cls, num, tt_dtype):
        res = super(AbstractInt, cls).__new__(cls, num)
        res._tt_dtype = tt_dtype
        res.num = num
        return res


class AbstractFloat(AbstractValue, float):
    def __new__(cls, num, tt_dtype):
        res = super(AbstractFloat, cls).__new__(cls, num)
        res._tt_dtype = tt_dtype
        res.num = num
        return res


class AbstractBool(AbstractValue):
    def __init__(self, num, tt_dtype):
        self._tt_dtype = tt_dtype
        self.num = num


@dataclass
class DummyCudaDevice:
    index: int


class AbstractPtr(AbstractValue):
    def __init__(self, tt_dtype: str, device: DummyCudaDevice = DummyCudaDevice(0)):
        self._tt_dtype = tt_dtype
        self.device = device

    def data_ptr(self):
        return


TYPE_MAKERS = {
    "I": lambda ty: AbstractInt(ty, "i32"),
    "f": lambda ty: AbstractFloat(ty, "f32"),
    "B": lambda ty: AbstractBool(ty, "i1"),
    "f8": lambda ty: AbstractFloat(ty, "f8"),
    "f16": lambda ty: AbstractFloat(ty, "f16"),
    "bf16": lambda ty: AbstractFloat(ty, "bf16"),
    "f32": lambda ty: AbstractFloat(ty, "f32"),
    "f64": lambda ty: AbstractFloat(ty, "f64"),
    "i1": lambda ty: AbstractBool(ty, "i1"),
    "i8": lambda ty: AbstractInt(ty, "i8"),
    "i16": lambda ty: AbstractInt(ty, "i16"),
    "i32": lambda ty: AbstractInt(ty, "i32"),
    "i64": lambda ty: AbstractInt(ty, "i64"),
}
