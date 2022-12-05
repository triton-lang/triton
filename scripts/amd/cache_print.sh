#!/bin/bash

CACHED_FILES=$(find /root/.triton/cache/ -type f -name "*.*")

rm -rf triton_cache
mkdir -p triton_cache

for file in ${CACHED_FILES[@]}; do
	echo "$file"
	if [[ $file == *.so ]]; then
		echo "Skipping printing .so file"
	elif [[ $file == *.cubin ]]; then
		echo "Skipping printing .cubin file"
	else
		sed -i -e '$a\' $file
		cat $file
		cp $file triton_cache
	fi
done
