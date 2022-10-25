#!/bin/bash

CACHED_FILES=$(find /root/.triton/cache/ -type f -name "*.*")

for file in ${CACHED_FILES[@]}; do
	echo "$file"
	if [[ $file == *.so ]]; then
		echo "Skipping printing .so file"
	else
		sed -i -e '$a\' $file
		cat $file
	fi
done
