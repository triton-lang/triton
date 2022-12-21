#!/bin/bash

# clear
set -x

# log dir
ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log_$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

# check for backtrace
if [ "$1" == "backtrace" ]; then
	sudo apt install gdb -y

	# COMMAND="-m pytest --capture=tee-sys --verbose python/tests/test_elementwise.py::test_single_input[log-float64-float64]"
	COMMAND="python/tutorials/05-layer-norm.py"
	gdb python \
		-ex "set pagination off" \
		-ex "run $COMMAND" \
		-ex "backtrace" \
		-ex "set confirm off" \
		-ex "q" \
		2>&1 | tee $LOG_DIR/backtrace.log

else

	sh scripts/amd/clean.sh

	# pytest -rfs --verbose python/tests 2>&1 | tee $LOG_DIR/test_all.log
	# pytest -rfs --verbose "python/tests/test_compiler.py" 2>&1 | tee $LOG_DIR/test_compiler.log
	# pytest -rfs --verbose "python/tests/test_core_amd.py" 2>&1 | tee $LOG_DIR/test_core_amd.log
	# pytest -rfs --verbose "python/tests/test_core.py" 2>&1 | tee $LOG_DIR/test_core.log
	# pytest -rfs --verbose "python/tests/test_core.py::test_math_op" | tee $LOG_DIR/test_math_op.log
	# pytest -rfs --verbose "python/tests/test_core.py::test_reduce1d[min-float16-128]" | tee $LOG_DIR/test_reduce1d.log
	# pytest -rfs --verbose "python/tests/test_core.py::test_reduce1d" | tee $LOG_DIR/test_reduce1d.log
	# pytest -rfs --verbose "python/tests/test_core.py::test_reduce2d" | tee $LOG_DIR/test_reduce2d.log
	# pytest -rfs --verbose "python/tests/test_elementwise.py" 2>&1 | tee $LOG_DIR/test_elementwise.log
	# pytest -rfs --verbose "python/tests/test_elementwise.py::test_single_input[log-float64-float64]" 2>&1 | tee $LOG_DIR/test_single_input.log
	# pytest -rfs --verbose "python/tests/test_ext_elemwise.py" 2>&1 | tee $LOG_DIR/test_ext_elemwise.log
	# pytest -rfs --verbose "python/tests/test_gemm.py" 2>&1 | tee $LOG_DIR/test_gemm.log
	# pytest -rfs --verbose "python/tests/test_reduce.py" 2>&1 | tee $LOG_DIR/test_reduce.log
	# pytest -rfs --verbose "python/tests/test_transpose.py" 2>&1 | tee $LOG_DIR/test_transpose.log
	# pytest -rfs --verbose "python/tests/test_vecadd.py" 2>&1 | tee $LOG_DIR/test_vecadd.log

	# tutorials
	# python  python/tutorials/01-vector-add.py 2>&1 | tee $LOG_DIR/01-vector-add.log
	# python  python/tutorials/02-fused-softmax.py 2>&1 | tee $LOG_DIR/02-fused-softmax.log
	# python  python/tutorials/03-matrix-multiplication.py 2>&1 | tee $LOG_DIR/03-matrix-multiplication.log
	# python  python/tutorials/04-low-memory-dropout.py 2>&1 | tee $LOG_DIR/04-low-memory-dropout.log
	python python/tutorials/05-layer-norm.py 2>&1 | tee $LOG_DIR/05-layer-norm.log
	# python  python/tutorials/06-fused-attention.py 2>&1 | tee $LOG_DIR/06-fused-attention.log
fi
