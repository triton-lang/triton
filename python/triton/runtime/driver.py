from ..third_party import backends


def _create_driver():
    actives = [x.driver for x in backends.values() if x.driver.is_active()]
    if len(actives) != 1:
        raise RuntimeError(f"{len(actives)} active drivers ({actives}). There should only be one.")
    return actives[0]()


driver = _create_driver()
