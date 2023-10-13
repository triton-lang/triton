sudo apt install gdb -y

# export AMD_OCL_WAIT_COMMAND=1
# export AMD_LOG_LEVEL=3
# export HIP_LAUNCH_BLOCKING=1

ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

COMMAND="python/tests/test_elementwise.py::test_single_input[log-float64-float64]"
gdb -ex "set pagination off" \
    -ex "file python" \
    -ex "run -m pytest --capture=tee-sys --verbose $COMMAND" \
    -ex "backtrace" \
    -ex "set confirm off" \
    -ex "q" \
    2>&1 | tee $LOG_DIR/gdb_backtrace.log
