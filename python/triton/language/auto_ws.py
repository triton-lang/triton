from .._C.libtriton import ir
from triton.language.core import builtin, constexpr, _constexpr_to_value, base_value
import re


class Group(base_value):
    """
    Context manager to annotate operations into groups manually.
    """
    def __init__(self, name, _builder=None):
        self.name = 'nvws.' + _constexpr_to_value(name)
        self.builder = _builder

    def __enter__(self):
        if not self.builder.options.ignore_manual_groups:
            self.builder.enter_group(self.name)

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.builder.options.ignore_manual_groups:
            self.builder.exit_group(self.name)


def group(name, start, size, _module, _builder):
    name = _constexpr_to_value(name)
    start = _constexpr_to_value(start)
    size = _constexpr_to_value(size)
    assert re.fullmatch(r'[A-Za-z][A-Za-z0-9\-_]*', name), 'invalid group name'
    if not _builder.options.ignore_manual_groups:
        _module.create_group('nvws.' + name, start, size)
    return constexpr(Group(name, _builder=_builder))
