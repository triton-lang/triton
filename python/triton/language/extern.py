from . import core
from pathlib import Path
from typing import Any


class ExternalFunction:
	def __init__(self, name: str, path: Path, ret_type: Any, arg_types: list) -> None:
		super().__init__()
		self._name = name
		self._path = path
		self._ret_type = ret_type
		self._arg_types = arg_types

	@core.builtin
	def __call__(self, args: list, _builder=None) -> Any:
		if len(args) != len(self.arg_types):
			raise ValueError(f"length of input args does not match function {self.name}'s declaration. Expect {len(args)}, got {len(self.arg_types)}")

		for i, arg in enumerate(args):
			match = True
			if type(arg) is core.tensor:
				match = arg.dtype == self.arg_types[i]
			else:
				match = type(arg) == self.arg_types[i]
			if not match:
				raise ValueError(f"input arg type does not match function {self.name}'s declaration. Expect {self.arg_types[0]}, got {arg.dtype}")

		func = getattr(_builder, self.name)
		return core.tensor(func(*args), self.ret_type)

	@property
	def name(self) -> str:
		return self._name

	@property
	def path(self) -> str:
		return self._path

	@property
	def ret_type(self) -> Any:
		return self._ret_type

	@property
	def arg_types(self) -> Any:
		return self._arg_types
