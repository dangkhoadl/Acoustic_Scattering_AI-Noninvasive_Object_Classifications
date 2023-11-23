#!/usr/bin/env bash
#
set -u           # Detect unset variable
set -e           # Exit on non-zero return code from any command
set -o pipefail  # Exit if any of the commands in the pipeline will
                 # return non-zero return code

export CUDA_VISIBLE_DEVICES="0,1";
export exp="exp-1"

resnet=0
ast=0
hubert=0
wav2vec2_conformer=1

## Resnet
if [[ ${resnet} -eq 1 ]] ; then
    function resnet_eval() {
        dir=$1
        dataset=$(echo ${dir} | cut -d '/' -f3)

        echo "resnet fully supervised: ${dataset}"
        python main/resnet.py \
            "conf/resnet.yaml" \
            --train_csv "metadata/${exp}/${dataset}/train.csv" \
            --test_csv "metadata/${exp}/${dataset}/test.csv" \
            --exp_name "${exp}-resnet-${dataset}"
    }

    export -f resnet_eval;
    find metadata/${exp}/ -mindepth 1 -type d \
        | xargs --max-procs=1 -I'%' bash -c 'resnet_eval %'
fi

## AST
if [[ ${ast} -eq 1 ]] ; then
    function ast_eval() {
        dir=$1
        dataset=$(echo ${dir} | cut -d '/' -f3)

        echo "ast finetune: ${dataset}"
        python main/AST_finetuned.py \
            "conf/AST_finetuned.yaml" \
            --train_csv "metadata/${exp}/${dataset}/train.csv" \
            --test_csv "metadata/${exp}/${dataset}/test.csv" \
            --exp_name "${exp}-ast_finetuned-${dataset}"

        echo "ast fully supervised: ${dataset}"
        python main/AST_fully-supervised.py \
            "conf/AST_fully-supervised.yaml" \
            --train_csv "metadata/${exp}/${dataset}/train.csv" \
            --test_csv "metadata/${exp}/${dataset}/test.csv" \
            --exp_name "${exp}-ast_fullysupervised-${dataset}"
    }

    export -f ast_eval;
    find metadata/${exp}/ -mindepth 1 -type d \
        | xargs --max-procs=1 -I'%' bash -c 'ast_eval %'
fi

## Hubert
if [[ ${hubert} -eq 1 ]] ; then
    function hubert_eval() {
        dir=$1
        dataset=$(echo ${dir} | cut -d '/' -f3)

        echo "hubert finetune: ${dataset}"
        python main/Hubert_finetuned.py \
            "conf/Hubert_finetuned.yaml" \
            --train_csv "metadata/${exp}/${dataset}/train.csv" \
            --test_csv "metadata/${exp}/${dataset}/test.csv" \
            --exp_name "${exp}-hubert_finetuned-${dataset}"

        echo "hubert finetune transformer: ${dataset}"
        python main/Hubert_finetuned.py \
            "conf/Hubert_finetuned-transformer.yaml" \
            --train_csv "metadata/${exp}/${dataset}/train.csv" \
            --test_csv "metadata/${exp}/${dataset}/test.csv" \
            --exp_name "${exp}-hubert_finetunedtransformer-${dataset}"
    }

    export -f hubert_eval;
    find metadata/${exp}/ -mindepth 1 -type d \
        | xargs --max-procs=1 -I'%' bash -c 'hubert_eval %'
fi

## Wav2vec2-Conformer
if [[ ${wav2vec2_conformer} -eq 1 ]] ; then
    function wav2vec2conformer_eval() {
        dir=$1
        dataset=$(echo ${dir} | cut -d '/' -f3)

        echo "wav2vec2conformer finetune: ${dataset}"
        python main/wav2vec2conformer_finetuned.py \
            "conf/wav2vec2conformer_finetuned.yaml" \
            --train_csv "metadata/${exp}/${dataset}/train.csv" \
            --test_csv "metadata/${exp}/${dataset}/test.csv" \
            --exp_name "${exp}-wav2vec2conformer_finetuned-${dataset}"

        echo "wav2vec2conformer finetune transformer: ${dataset}"
        python main/wav2vec2conformer_finetuned.py \
            "conf/wav2vec2conformer_finetuned-transformer.yaml" \
            --train_csv "metadata/${exp}/${dataset}/train.csv" \
            --test_csv "metadata/${exp}/${dataset}/test.csv" \
            --exp_name "${exp}-wav2vec2conformer_finetunedtransformer-${dataset}"
    }

    export -f wav2vec2conformer_eval;
    find metadata/${exp}/ -mindepth 1 -type d \
        | xargs --max-procs=1 -I'%' bash -c 'wav2vec2conformer_eval %'
fi
