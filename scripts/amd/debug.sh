sudo apt install gdb -y

# export AMD_OCL_WAIT_COMMAND=1
# export AMD_LOG_LEVEL=3
# export HIP_LAUNCH_BLOCKING=1

gdb -ex "set pagination off" \
    -ex "file python" \
    -ex "set confirm off" \
    -ex "break 1" \
    -ex 'run -m pytest --capture=tee-sys --verbose "python/test/unit/language/test_core.py::test_load_and_store_op[float32-2]"' \
    -ex "q" \
    2>&1 | tee /dockerx/pytorch/test_core_gdb.log
