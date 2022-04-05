from dataclasses import dataclass


class AbstractValue:
    @property
    def tt_dtype(self):
        return self._tt_dtype

    @property
    def is_attr(self):
        return hasattr(self, "num")

    @property
    def val(self):
        if self.is_attr:
            return self.num

    def __eq__(self, o: object) -> bool:
        if self.is_attr:
            return self.num == o
        return self is o

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__.replace("Input", "")
        # TODO: new mangle function expects attributes to return numbers. i think code_gen.py is the place to fix that. meanwhile, this HACK is here
        if self.is_attr:
            return str(self.num)

        repr = f"<{self.tt_dtype}"
        if "Ptr" in cls_name:
            repr += "*"
        if self.is_attr:
            repr += f",{self.num}"
        repr += ">"
        return repr


class AbstractInt(AbstractValue, int):
    def __new__(cls, num, tt_dtype):
        """
        we need the tt_dtype here since we can have several integer sizes (i32,i64,...).
        we need to pass this information to triton compiler later on
        """
        res = super(AbstractInt, cls).__new__(cls, num)
        res._tt_dtype = tt_dtype
        res.num = num
        return res


class AbstractFloat(AbstractValue, float):
    def __new__(cls, num, tt_dtype):
        """
        we need the tt_dtype here since we can have several float sizes (f32,bf32,...).
        we need to pass this information to triton compiler later on
        """
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
        # TODO: triton does alignment optimization. We will make this return all possible alignments (maybe 4,8,16?)
        return 140010382688256


TYPE_MAKERS = {
    "I": lambda ty: AbstractInt(ty, "i32"),
    "L": lambda ty: AbstractInt(ty, "i64"),
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
    "u8": lambda ty: AbstractInt(ty, "u8"),
    "u16": lambda ty: AbstractInt(ty, "u16"),
    "u32": lambda ty: AbstractInt(ty, "u32"),
    "u64": lambda ty: AbstractInt(ty, "u64"),
}


TRITON_TO_C_TYPES = {
    "I": "int32_t",
    "L": "int64_t",
    "f": "float",
    "B": "bool",
    "f8": "float",
    "f16": "float",
    "bf16": "float",
    "f32": "float",
    "f64": "double",
    "i1": "bool",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "unit32_t",
    "u64": "uint64_t",
}
