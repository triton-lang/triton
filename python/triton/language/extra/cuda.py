import os

from .. import core

__path__ = os.path.dirname(os.path.abspath(__file__))


@core.extern
def globaltimer(_builder=None):
    return core.elementwise("extra", os.path.join(__path__, "extra.bc"), [],
                            {tuple(): ("globaltimer", core.dtype("int64")),
                             }, _builder)
