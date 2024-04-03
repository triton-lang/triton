from triton.runtime.driver import driver

target = driver.active.get_current_target()[0]
if target == "cuda":
    import triton.language.extra.cuda.libdevice as libdevice
if target == "hip":
    import triton.language.extra.hip.libdevice as libdevice

# Import all the attributes from the selected libdevice module into the current module's namespace
globals().update(vars(libdevice))
