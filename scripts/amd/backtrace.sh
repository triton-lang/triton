sudo apt install gdb -y

# export AMD_OCL_WAIT_COMMAND=1
# export AMD_LOG_LEVEL=3
# export HIP_LAUNCH_BLOCKING=1


gdb -ex "set pagination off" \
    -ex "file python" \
    -ex 'run -m pytest --capture=tee-sys --verbose "python/test/unit/language/test_core.py::test_dot[none-False-float16]"' \
    -ex "backtrace" \
    -ex "set confirm off" \
    -ex "q" \
    2>&1 | tee /dockerx/pytorch/test_core_gdb.log
