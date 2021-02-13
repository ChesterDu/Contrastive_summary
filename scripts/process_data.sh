#!/bin/bash
dataset=$1
seed=$2
processed_data_dir="processed_data"
if [ ! -d "${processed_data_dir}" ]; then
    mkdir -p "${processed_data_dir}"
fi
if [ ! -d "${processed_data_dir}/${dataset}" ]; then
    mkdir -p "${processed_data_dir}/${dataset}"
    mkdir -p "${processed_data_dir}/${dataset}/train"
    mkdir -p "${processed_data_dir}/${dataset}/test"
fi

python src/sample_data.py --dataset $dataset --seed $seed
echo "==========================="
echo "  begin generate summary   "
echo "==========================="
cd PreSumm/src
python train.py -task abs -mode test_text -batch_size 100 -test_batch_size 100 -text_src ../raw_data/"$dataset".txt -test_from ../models/model_step_148000.pt -visible_gpus 0 -result_path ../results/"$dataset" -log_file ../logs/"$dataset".log
cd ../..
cp "PreSumm/results/${dataset}.-1.candidate" "processed_data/${dataset}/train/summary"
