sudo apt install gdb -y

# export AMD_OCL_WAIT_COMMAND=1
# export AMD_LOG_LEVEL=3
# export HIP_LAUNCH_BLOCKING=1

gdb -ex "file python" \
    -ex 'run -m pytest --capture=tee-sys --verbose "python/test/unit/language/test_core.py::test_empty_kernel[float32]"' \
    -ex "set pagination off" \
    -ex "set confirm off" \
    -ex "break _exit" \
    -ex "commands"
    -ex "run"
    -ex 'end' \
    2>&1 | tee /dockerx/pytorch/test_core_gdb.log
