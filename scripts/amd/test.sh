#!/bin/bash

# clear
set -x


# log dir
ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log_$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

sh scripts/amd/clean.sh

UNIT_TEST="python/test/unit/language/test_core_amd.py"
# UNIT_TEST="python/test/unit/language/test_core.py::test_empty_kernel[float32]"
# UNIT_TEST="python/test/unit/runtime/test_cache.py::test_compile_in_subproc"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_shift_op[int8-int8-<<]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_shift_op[int32-int32->>]"
# UNIT_TEST="python/test/unit/language/test_core.py::test_bin_op"
# UNIT_TEST="python/test/unit/language/test_core.py::test_bin_op[float32-float32-+]"
# UNIT_TEST="python/test/unit/language/test_core.py::test_bin_op[int8-float16-%]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_masked_load_shared_memory[dtype0]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_masked_load_shared_memory[dtype1]"

# check for backtrace
if [ "$1" == "backtrace" ]; then
	sudo apt install gdb -y

	COMMAND="-m pytest --capture=tee-sys --verbose $UNIT_TEST"
	gdb python \
		-ex "set pagination off" \
		-ex "run $COMMAND" \
		-ex "backtrace" \
		-ex "set confirm off" \
		-ex "q" \
		2>&1 | tee $LOG_DIR/backtrace.log

else
	pytest --capture=tee-sys -rfs --verbose "$UNIT_TEST" 2>&1 | tee $LOG_DIR/unit_test.log
fi

# bash scripts/amd/cache_print.sh  2>&1 |tee $LOG_DIR/cache.log
