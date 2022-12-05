sudo apt install gdb -y

# export AMD_OCL_WAIT_COMMAND=1
# export AMD_LOG_LEVEL=3
# export HIP_LAUNCH_BLOCKING=1

ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log_$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

gdb -ex "set pagination off" \
    -ex "file python" \
    -ex 'run -m pytest --capture=tee-sys --verbose "python/tests/test_vecadd.py::test_vecadd_no_scf[1-64-shape0]"' \
    -ex "backtrace" \
    -ex "set confirm off" \
    -ex "q" \
    2>&1 | tee $LOG_DIR/gdb_backtrace.log
