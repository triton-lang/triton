import os

from .. import core

__path__ = os.path.dirname(os.path.abspath(__file__))


@core.extern
def globaltimer(_builder=None):
    return core.extern_elementwise("cuda", os.path.join(__path__, "cuda.bc"), [],
                                   {tuple(): ("globaltimer", core.dtype("int64")),
                                    }, is_pure=False, _builder=_builder)
