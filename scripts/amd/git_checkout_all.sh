#!/bin/bash
set -x

CUR_BRANCH=$(git rev-parse --abbrev-ref HEAD)
REF_BRANCH=$1

DIFF_FILES=$(git diff --name-only $REF_BRANCH $CUR_BRANCH)
for file in $DIFF_FILES; do
	echo "$file"
	git checkout $REF_BRANCH $file
done
