from __future__ import annotations

import operator
from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from types import NotImplementedType
from typing import Any, Callable, ClassVar, Protocol, final, overload, override

from .tensor_metadata import TensorMetadata

# precedence:
# 0: paren
# 1: not (~)
# 2: and (&)
# 3: or (|)
# 4: comparisons
# (note that because we use bitwise operators, the precedence is *lower* than comparisons)


@dataclass
class CheckResult:
    ok: bool = True
    reason: str = ""

    def __bool__(self):
        return self.ok

    def __str__(self):
        return self.reason


def _fail(reason: str) -> CheckResult:
    return CheckResult(ok=False, reason=reason)


def _ok() -> CheckResult:
    return CheckResult()


@dataclass
class TypeAssertion:
    _precedence: ClassVar[int] = 999

    @final
    def is_valid(self, t: TensorMetadata) -> CheckResult:
        r = self._is_valid(t)
        if isinstance(r, bool):
            r = CheckResult(ok=r, reason=str(self))
        return r

    def _is_valid(self, t: TensorMetadata) -> bool | CheckResult:
        return False

    @overload
    def __and__(self, other: TypeAssertion) -> TypeAssertion:
        ...

    @overload
    def __and__(self, other: object) -> NotImplementedType:
        ...

    def __and__(self, other: object) -> TypeAssertion | NotImplementedType:
        if not isinstance(other, TypeAssertion):
            return NotImplemented

        return And(left=self, right=other)

    @overload
    def __or__(self, other: TypeAssertion) -> TypeAssertion:
        ...

    @overload
    def __or__(self, other: object) -> NotImplementedType:
        ...

    def __or__(self, other: object) -> TypeAssertion | NotImplementedType:
        if not isinstance(other, TypeAssertion):
            return NotImplemented

        return Or(left=self, right=other)

    def __invert__(self) -> TypeAssertion:
        return Not(self)

    def _format(self, other: TypeAssertion, *, right: bool = False) -> str:
        other_prec = other._precedence + int(right)
        return f"({other})" if other_prec > self._precedence else f"{other}"


@dataclass
class Unary(TypeAssertion):
    item: TypeAssertion


@dataclass
class Not(Unary):
    _precedence: ClassVar[int] = 1

    @override
    def _is_valid(self, t: TensorMetadata) -> CheckResult:
        if isinstance(self.item, Not):
            return self.item.item.is_valid(t)
        else:
            r = self.item.is_valid(t)
            return CheckResult(ok=r.ok, reason="not " + r.reason)

    def __str__(self) -> str:
        return "~" + self._format(self.item)


@dataclass
class Binary(TypeAssertion):
    left: TypeAssertion
    right: TypeAssertion
    _op_str: ClassVar[str]

    def __str__(self) -> str:
        return f"{self._format(self.left)} {self._op_str} {self._format(self.right, right=True)}"


@dataclass
class And(Binary):
    _precedence: ClassVar[int] = 2
    _op_str: ClassVar[str] = "&"

    @override
    def _is_valid(self, t: TensorMetadata) -> CheckResult:
        return self.left.is_valid(t) and self.right.is_valid(t)

    def __str__(self) -> str:
        return self._format(self.left) + " & " + self._format(self.right, right=True)


@dataclass
class Or(Binary):
    _precedence: ClassVar[int] = 3
    _op_str: ClassVar[str] = "|"

    @override
    def _is_valid(self, t: TensorMetadata) -> CheckResult:
        return self.left.is_valid(t) or self.right.is_valid(t)


@dataclass
class Atom(TypeAssertion):
    _precedence: ClassVar[int] = 0


@dataclass
class Unsharded(Atom):

    @override
    def _is_valid(self, t: TensorMetadata) -> bool:
        return t.tensor_sharding is None

    def __str__(self) -> str:
        return "Unsharded()"


@dataclass
class Sharded(Atom):
    dim: int
    _: KW_ONLY
    uniform: bool = False

    @override
    def _is_valid(self, t: TensorMetadata) -> bool:
        s = t.tensor_sharding
        return (s is not None and self.dim == s.dim and not (self.uniform and s.uniform_width is None))

    def __str__(self) -> str:
        parts = [f"{self.dim}"]
        if self.uniform:
            parts.append(f"uniform={self.uniform}")
        return f"Sharded({', '.join(parts)})"


class Op(Protocol):
    TYPES: dict[str, TypeAssertion]


@dataclass
class OpInput(Atom):
    op: Op
    name: str

    _nested: TypeAssertion = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._nested = self.op.TYPES[self.name]

    @override
    def _is_valid(self, t: TensorMetadata) -> bool:
        return self._nested.is_valid(t)


@dataclass
class Dim(Atom):
    n: int

    def _is_valid(self, t: TensorMetadata) -> bool:
        return len(t.shape) == self.n

    def __str__(self) -> str:
        return f"Dim({self.n})"


@dataclass
class _Size:

    @abstractmethod
    def get(self, t: TensorMetadata) -> int:
        ...

    @overload
    def __lt__(self, other: _Size) -> TypeAssertion:
        ...

    @overload
    def __lt__(self, other: int) -> TypeAssertion:
        ...

    @overload
    def __lt__(self, other: object) -> NotImplementedType:
        ...

    def __lt__(self, other: object) -> TypeAssertion | NotImplementedType:
        return _make_comparison(self, other, "<", operator.__lt__)

    @overload
    def __le__(self, other: _Size) -> TypeAssertion:
        ...

    @overload
    def __le__(self, other: int) -> TypeAssertion:
        ...

    @overload
    def __le__(self, other: object) -> NotImplementedType:
        ...

    def __le__(self, other: object) -> TypeAssertion | NotImplementedType:
        return _make_comparison(self, other, "<=", operator.__le__)

    @overload
    def __gt__(self, other: _Size) -> TypeAssertion:
        ...

    @overload
    def __gt__(self, other: int) -> TypeAssertion:
        ...

    @overload
    def __gt__(self, other: object) -> NotImplementedType:
        ...

    def __gt__(self, other: object) -> TypeAssertion | NotImplementedType:
        return _make_comparison(self, other, ">", operator.__gt__)

    @overload
    def __ge__(self, other: _Size) -> TypeAssertion:
        ...

    @overload
    def __ge__(self, other: int) -> TypeAssertion:
        ...

    @overload
    def __ge__(self, other: object) -> NotImplementedType:
        ...

    def __ge__(self, other: object) -> TypeAssertion | NotImplementedType:
        return _make_comparison(self, other, ">=", operator.__ge__)

    @overload
    def __eq__(self, other: _Size) -> TypeAssertion:
        ...

    @overload
    def __eq__(self, other: int) -> TypeAssertion:
        ...

    @overload
    def __eq__(self, other: object) -> NotImplementedType:
        ...

    def __eq__(self, other: object) -> TypeAssertion | NotImplementedType:
        return _make_comparison(self, other, "==", operator.__eq__)

    @overload
    def __ne__(self, other: _Size) -> TypeAssertion:
        ...

    @overload
    def __ne__(self, other: int) -> TypeAssertion:
        ...

    @overload
    def __ne__(self, other: object) -> NotImplementedType:
        ...

    def __ne__(self, other: object) -> TypeAssertion | NotImplementedType:
        return _make_comparison(self, other, "!=", operator.__ne__)


@dataclass
class _SizeComparison(TypeAssertion):
    left: _Size
    right: _Size
    op_str: str
    op: Callable[[Any, Any], bool]

    _precedence: ClassVar[int] = 4

    def _is_valid(self, t: TensorMetadata) -> bool:
        left = self.left.get(t)
        right = self.right.get(t)
        return left is not None and right is not None and self.op(left, right)

    def __str__(self) -> str:
        return f"{self.left} {self.op_str} {self.right}"


def _make_comparison(left: _Size, right: object, op_str: str, op: Callable[[Any, Any],
                                                                           bool]) -> TypeAssertion | NotImplementedType:
    if isinstance(right, int):
        right = _ConstantSize(right)
    elif not isinstance(right, _Size):
        return NotImplemented
    return _SizeComparison(left, right, op_str, op)


@dataclass
class _ConstantSize(_Size):
    n: int

    @override
    def get(self, t: TensorMetadata) -> int | None:
        return self.n

    def __str__(self) -> str:
        return str(self.n)


@dataclass
class _SizeIsStatic(Atom):
    n: int

    @override
    def _is_valid(self, t: TensorMetadata) -> bool:
        return self.n not in t.dynamic_dims

    def __str__(self) -> str:
        return f"Size({self.n}).is_static"


@dataclass
class Size(_Size):
    n: int

    @override
    def get(self, t: TensorMetadata) -> int | None:
        is_dynamic = self.n in t.dynamic_dims
        return None if is_dynamic else t.shape[self.n]

    @property
    def is_dynamic(self) -> TypeAssertion:
        return Not(self.is_static)

    @property
    def is_static(self) -> TypeAssertion:
        return _SizeIsStatic(self.n)

    def __str__(self):
        return f"Size({self.n})"
