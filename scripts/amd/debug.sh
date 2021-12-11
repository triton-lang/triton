gdb -ex "set pagination off" \
    -ex "file python" \
    -ex 'run -m pytest --capture=tee-sys --verbose "python/test/unit/language/test_core.py::test_load_and_store_op[float64]"' \
    -ex "bt" \
    -ex "set confirm off" \
    -ex "q" \
    2>&1 | tee /dockerx/pytorch/test_core_gdb.log
