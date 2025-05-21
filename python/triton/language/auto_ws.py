from .._C.libtriton import ir
from triton.language.core import builtin, constexpr, _constexpr_to_value, base_value
import re


class Group(base_value):
    """
    Context manager to annotate operations into groups manually.
    """
    def __init__(self, name):
        self.name = 'nvws.' + _constexpr_to_value(name)
        self.builder = None

    def set_builder(self, builder):
        self.builder = builder

    def clear_builder(self):
        self.builder = None

    def __enter__(self):
        assert self.builder is not None
        if not self.builder.options.ignore_manual_groups:
            self.builder.enter_group(self.name)

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.builder.options.ignore_manual_groups:
            self.builder.exit_group(self.name)


def group(name, start, size, reg_count=None, _module=None, _builder=None):
    name = _constexpr_to_value(name)
    start = _constexpr_to_value(start)
    size = _constexpr_to_value(size)
    if reg_count is not None:
        reg_count = _constexpr_to_value(reg_count)
    else:
        reg_count = 0
    assert re.fullmatch(r'[A-Za-z][A-Za-z0-9\-_]*', name), 'invalid group name'
    if not _builder.options.ignore_manual_groups:
        _module.create_group('nvws.' + name, start, size, reg_count)
    return constexpr(Group(name))
