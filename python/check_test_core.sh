#! /bin/bash

TEST_OUTPUT=test_output
cd $(git rev-parse --show-toplevel)/python

if [ ! -d $TEST_OUTPUT ];then
    mkdir $TEST_OUTPUT
fi

TEST_CORE=./test/unit/language/test_core.py
pytest --co -q ${TEST_CORE} | sed "s/\[.*\]//g" | awk '!a[$0]++' | sed -n "/test_core.py/p" > test_names.txt

BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT=$(git rev-parse --short HEAD)

SUMMARY=${TEST_OUTPUT}/summary_${BRANCH}@${COMMIT}_$(date '+%m-%d-%Y-%H-%M-%S').txt

echo "Running test_core.py on ${BRANCH}@${COMMIT}" | tee -a ${SUMMARY}

while read -r line; do
    test_name=${line##*::}
    if [[ ${line:0:1} != "#" ]];then
        printf '%-45s' "${test_name}: " | tee -a $SUMMARY
        Msg=$(pytest -v $line > ${TEST_OUTPUT}/${test_name}.output 2>&1)
        if [ $? -ne 0 ] && grep -Fq "Aborted" ${TEST_OUTPUT}/${test_name}.output; then
            printf "Aborted\n" | tee -a $SUMMARY
        else
            last_line=$(sed -n '/====/p' ${TEST_OUTPUT}/${test_name}.output | tail -n 1)
            report=${last_line#=* }
            report=${report%%in *}
            printf "$report \n" | tee -a $SUMMARY
        fi
    fi
done < test_names.txt



