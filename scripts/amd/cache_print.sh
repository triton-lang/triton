#!/bin/bash

CACHED_FILES=$(find /root/.triton/cache/ -type f -name "*.*")

for file in ${CACHED_FILES[@]}; do
	echo "$file"
	sed -i -e '$a\' $file
	cat $file
done
