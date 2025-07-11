#!/bin/bash

# This script is used to demonstrate the use of the TTGIR instrumentation feature in Proton.
# the IR may change over time so to make this script future proof we will
# instrument at the DSL level first and then use that IR to override the uninstrumented kernel
DUMP_DIR="$PWD/ttgir_dump"

if [ -e "$DUMP_DIR" ];
	then rm -rf "$DUMP_DIR" ;
fi

mkdir -p "$DUMP_DIR"

TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=$DUMP_DIR python vector-add-instrumented.py
# Iterate over all subdirectories in $DUMP_DIR and remove all except the .ttgir files
for dir in "$DUMP_DIR"/*; do
	if [ -d "$dir" ]; then
		find "$dir" -type f ! -name "*.ttgir" -delete
		#Save off the actual hash directory (this will change across kernel/Triton/etc. versions)
		TTGIR_DIR="$dir"
	fi
done

echo "TTGIR files dumped to $TTGIR_DIR"

# Save the add_kernel.ttgir file from the DSL level instrumentation to the current directory temporarily
cp $TTGIR_DIR/add_kernel.ttgir $PWD
rm -rf "$DUMP_DIR"

# Now run the uninstrumented kernel and overwrite the add_kernel.ttgir file from the DSL level instrumentation
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=$DUMP_DIR python vector-add.py

for dir in "$DUMP_DIR"/*; do
	if [ -d "$dir" ]; then
		find "$dir" -type f ! -name "*.ttgir" -delete
		TTGIR_DIR="$dir"
	fi
done

mv add_kernel.ttgir $TTGIR_DIR/add_kernel.ttgir

TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=$DUMP_DIR python vector-add.py

echo "Now run `proton-viewer -m normalized_cycles vector-add.hatchet` to see the output"
