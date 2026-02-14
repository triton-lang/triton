import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, ParamSpec, TypeVar

from .tensor import Tensor
from .tensor_types import TypeAssertion


def _format_callable(func: Any) -> str:
    unwrapped = inspect.unwrap(func)
    module = inspect.getmodule(unwrapped)
    qualname = getattr(unwrapped, "__qualname__", repr(unwrapped))
    if module is None:
        return qualname
    return f"{module.__name__}.{qualname}"


P = ParamSpec("P")
R = TypeVar("R")


@dataclass(kw_only=True)
class Op:
    # Type assertions for all tensor arguments; required for all tensor arguments.
    types: dict[str, TypeAssertion] = field(default_factory=dict)

    def __call__(self, forward: Callable[P, R]) -> Callable[P, R]:

        def apply(*args: P.args, **kwargs: P.kwargs) -> R:
            """Type-check and forward to wrapped function."""
            func_name: str | None = None

            def get_func_name() -> str:
                nonlocal func_name
                if func_name is None:
                    func_name = _format_callable(forward)
                return func_name

            sig = inspect.signature(forward)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for name, value in bound_args.arguments.items():
                is_tensor = isinstance(value, Tensor)
                type_assertion = self.types.get(name)

                if is_tensor:
                    if type_assertion is None:
                        raise ValueError(f"{get_func_name()}: parameter {name} is a tensor, but has no type assertion")
                    check_result = type_assertion.is_valid(value)
                    if not check_result:
                        raise ValueError(f"{get_func_name()}: parameter {name}: type assertion failed: {check_result}")
                elif type_assertion is not None:
                    raise ValueError(
                        f"{get_func_name()}: parameter {name} has type assertion, but is not a tensor (it is {type(value)})"
                    )

            return forward(*bound_args.args, **bound_args.kwargs)

        return apply
