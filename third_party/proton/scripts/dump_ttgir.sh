#!/bin/bash
# Usage: ./dump_ttgir.sh python <your_script.py>

cmd="$*"
if [ -z "$cmd" ]; then
	echo "Example usage: $0 python <your_script.py>"
	exit 1
fi

DUMP_DIR="$PWD/ttgir_dump"
mkdir -p "$DUMP_DIR"

TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=$DUMP_DIR $cmd
# Iterate over all subdirectories in $DUMP_DIR and remove all except the .ttgir files
for dir in "$DUMP_DIR"/*; do
	if [ -d "$dir" ]; then
		find "$dir" -type f ! -name "*.ttgir" -delete
	fi
done

echo "TTGIR files dumped to $DUMP_DIR"
