from triton.runtime import driver

__all__ = ["current_target"]


def current_target(_semantic=None):
    return driver.active.get_current_target()


current_target.__triton_builtin__ = True
