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

# check for backtrace
if [ "$1" == "backtrace" ]; then
	sudo apt install gdb -y

	COMMAND="-m pytest --capture=tee-sys --verbose python/tests/test_core_amd.py::test_bin_op[int8-int8-/]"
	# COMMAND="python/tutorials/05-layer-norm.py"
	gdb python \
		-ex "set pagination off" \
		-ex "run $COMMAND" \
		-ex "backtrace" \
		-ex "set confirm off" \
		-ex "q" \
		2>&1 | tee $LOG_DIR/backtrace.log

else
	# fs --verbose python/test 2>&1 | tee $LOG_DIR/test_all.log
	# pytest -rfs --verbose python/test/unit 2>&1 | tee $LOG_DIR/test_unit.log
	# pytest -rfs --verbose python/test/unit/language 2>&1 | tee $LOG_DIR/language.log
	# pytest -rfs --verbose python/test/unit/language/test_core.py 2>&1 | tee $LOG_DIR/language.log
	# pytest -rfs --verbose python/test/unit/language/test_core_amd.py 2>&1 | tee $LOG_DIR/language.log
	# pytest -rfs --verbose python/test/unit/language/test_printf.py 2>&1 | tee $LOG_DIR/language.log
	# pytest -rfs --verbose python/test/unit/language/test_random.py 2>&1 | tee $LOG_DIR/language.log
	# pytest -rfs --verbose python/test/unit/operators 2>&1 | tee $LOG_DIR/operators.log
	# pytest -rfs --verbose python/test/unit/operators/test_blocksparse.py 2>&1 | tee $LOG_DIR/operators.log
	# pytest -rfs --verbose python/test/unit/operators/test_cross_entropy.py 2>&1 | tee $LOG_DIR/operators.log
	# pytest -rfs --verbose python/test/unit/operators/test_matmul.py 2>&1 | tee $LOG_DIR/operators.log
	# pytest -rfs --verbose python/test/unit/runtime 2>&1 | tee $LOG_DIR/runtime.log
	# pytest -rfs --verbose python/test/regression 2>&1 | tee $LOG_DIR/test_regression.logpytest -r
	
	# pytest -rfs --verbose "python/tests/test_core_amd.py" 2>&1 | tee $LOG_DIR/test_core_amd.log
	# pytest -rfs --verbose "python/tests/test_core_amd.py::test_bitwise_op"
	# pytest -rfs --verbose "python/tests/test_core_amd.py::test_bitwise_op[int8-int8-&1]"
	# pytest -rfs --verbose "python/tests/test_core_amd.py::test_bin_op[int8-int8-/]"
	# pytest -rfs --verbose "python/tests/test_core_amd.py::test_empty_kernel" 2>&1 | tee $LOG_DIR/test_empty_kernel.log
	# pytest -rfs --verbose "python/tests/test_core_amd.py::test_empty_kernel[float32]" 2>&1 | tee $LOG_DIR/test_empty_kernel_float32.log
	# pytest -rfs --verbose "python/tests/test_core_amd.py::test_bin_op[float32-float32-+]" 2>&1 | tee $LOG_DIR/test_bin_op_float32.log
	# pytest -rfs --verbose "python/test/unit/language/test_core.py" 2>&1 | tee $LOG_DIR/test_core_amd.log
	# pytest -rfs --verbose "python/tests/test_core.py" 2>&1 | tee $LOG_DIR/test_core.log
	# pytest -rfs --verbose "python/tests/test_core.py::test_math_op" | tee $LOG_DIR/test_math_op.log
	# pytest -rfs --verbose "python/tests/test_core.py::test_reduce1d[min-float16-128]" | tee $LOG_DIR/test_reduce1d.log
	# pytest -rfs --verbose "python/tests/test_core.py::test_reduce1d" | tee $LOG_DIR/test_reduce1d.log
	# pytest -rfs --verbose "python/tests/test_core.py::test_reduce2d" | tee $LOG_DIR/test_reduce2d.log

	pytest -rfs --verbose "python/test/unit/language/test_compiler.py" 2>&1 | tee $LOG_DIR/test_compiler.log
	pytest -rfs --verbose "python/test/unit/language/test_core_amd.py" 2>&1 | tee $LOG_DIR/test_core_amd.log
	# pytest -rfs --verbose "python/test/unit/language/test_core.py" 2>&1 | tee $LOG_DIR/test_core.log
	pytest -rfs --verbose "python/test/unit/language/test_elementwise.py" 2>&1 | tee $LOG_DIR/test_elementwise.log
	pytest -rfs --verbose "python/test/unit/language/test_ext_elemwise.py" 2>&1 | tee $LOG_DIR/test_ext_elemwise.log
	pytest -rfs --verbose "python/test/unit/language/test_gemm.py" 2>&1 | tee $LOG_DIR/test_gemm.log
	pytest -rfs --verbose "python/test/unit/language/test_printf.py" 2>&1 | tee $LOG_DIR/test_printf.log
	pytest -rfs --verbose "python/test/unit/language/test_reduce.py" 2>&1 | tee $LOG_DIR/test_reduce.log
	pytest -rfs --verbose "python/test/unit/language/test_transpose.py" 2>&1 | tee $LOG_DIR/test_transpose.log
	pytest -rfs --verbose "python/test/unit/language/test_vecadd.py" 2>&1 | tee $LOG_DIR/test_vecadd.log

	# tutorials
	# python  python/tutorials/01-vector-add.py 2>&1 | tee $LOG_DIR/01-vector-add.log
	# python  python/tutorials/02-fused-softmax.py 2>&1 | tee $LOG_DIR/02-fused-softmax.log
	# python  python/tutorials/03-matrix-multiplication.py 2>&1 | tee $LOG_DIR/03-matrix-multiplication.log
	# python  python/tutorials/04-low-memory-dropout.py 2>&1 | tee $LOG_DIR/04-low-memory-dropout.log
	# python python/tutorials/05-layer-norm.py 2>&1 | tee $LOG_DIR/05-layer-norm.log
	# python  python/tutorials/06-fused-attention.py 2>&1 | tee $LOG_DIR/06-fused-attention.log
fi
