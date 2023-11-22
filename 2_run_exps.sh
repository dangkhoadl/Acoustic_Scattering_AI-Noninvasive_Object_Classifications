#!/usr/bin/env bash
#
set -u           # Detect unset variable
set -e           # Exit on non-zero return code from any command
set -o pipefail  # Exit if any of the commands in the pipeline will
                 # return non-zero return code

export CUDA_VISIBLE_DEVICES="0,1,2,3";


resnet=1


if [[ ${resnet} -eq 1 ]] ; then
    function resnet_eval() {
        dir=$1
        dataset=$(echo ${dir} | cut -d '/' -f3)

        echo "resnet fully supervised: ${dataset}"
        python main/resnet.py \
            "conf/resnet.yaml" \
            --train_csv "data/exp-1/${dataset}/train.csv" \
            --test_csv "data/exp-1/${dataset}/test.csv" \
            --exp_name "exp-1-${dataset}"
    }

    export -f resnet_eval;
    find data/exp-1/ -mindepth 1 -type d \
        | xargs --max-procs=1 -I'%' bash -c 'resnet_eval %'
fi
