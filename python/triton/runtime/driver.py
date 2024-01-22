from ..backends import backends
from ..backends import DriverBase


def _create_driver():
    actives = [x.driver for x in backends.values() if x.driver.is_active()]
    if len(actives) != 1:
        raise RuntimeError(f"{len(actives)} active drivers ({actives}). There should only be one.")
    return actives[0]()


class LazyProxy:

    def __init__(self, init_fn):
        self._init_fn = init_fn
        self._obj = None

    def _initialize_obj(self):
        if self._obj is None:
            self._obj = self._init_fn()

    def __getattr__(self, name):
        self._initialize_obj()
        return getattr(self._obj, name)

    def __setattr__(self, name, value):
        if name in ["_init_fn", "_obj"]:
            super().__setattr__(name, value)
        else:
            self._initialize_obj()
            setattr(self._obj, name, value)

    def __delattr__(self, name):
        self._initialize_obj()
        delattr(self._obj, name)

    def __repr__(self):
        if self._obj is None:
            return f"<{self.__class__.__name__} for {self._init_fn} not yet initialized>"
        return repr(self._obj)

    def __str__(self):
        self._initialize_obj()
        return str(self._obj)


class DriverConfig:

    def __init__(self):
        self.default = LazyProxy(_create_driver)
        self.active = self.default

    def set_active(self, driver: DriverBase):
        self.active = driver

    def reset_active(self):
        self.active = self.default


driver = DriverConfig()
