from __future__ import annotations

import os

from ..backends import backends, DriverBase


def _create_driver() -> DriverBase:
    selected = os.environ.get("TRITON_DEFAULT_BACKEND", None)
    if selected:
        if selected not in backends:
            raise RuntimeError(f"Unknown backend device '{selected}'. Available backends: {list(backends.keys())}")
        driver = backends[selected].driver
        if not driver.is_active():
            raise RuntimeError(f"Backend device '{selected}' is not active.")
        return driver()
    else:
        active_drivers = [x.driver for x in backends.values() if x.driver.is_active()]
        if len(active_drivers) != 1:
            raise RuntimeError(f"{len(active_drivers)} active drivers ({active_drivers}). There should only be one.")
        return active_drivers[0]()


class DriverConfig:

    def __init__(self) -> None:
        self._default: DriverBase | None = None
        self._active: DriverBase | None = None

    @property
    def default(self) -> DriverBase:
        if self._default is None:
            self._default = _create_driver()
        return self._default

    @property
    def active(self) -> DriverBase:
        if self._active is None:
            self._active = self.default
        return self._active

    def set_active(self, driver: DriverBase) -> None:
        self._active = driver

    def reset_active(self) -> None:
        self._active = self.default


driver = DriverConfig()
