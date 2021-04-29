#!/bin/bash
dataset=$1
seed=$2
processed_data_dir="processed_data"
if [ ! -d "${processed_data_dir}" ]; then
    mkdir -p "${processed_data_dir}"
fi
# if [ ! -d "${processed_data_dir}/${dataset}" ]; then
mkdir -p "${processed_data_dir}/${dataset}"
mkdir -p "${processed_data_dir}/${dataset}/seed-${seed}"
mkdir -p "${processed_data_dir}/${dataset}/seed-${seed}/train"
mkdir -p "${processed_data_dir}/${dataset}/seed-${seed}/test"
# fi

echo "==========================="
echo "  begin Sample Data   "
echo "==========================="

python src/sample_data.py --dataset $dataset --seed $seed
echo "==========================="
echo "  begin generate summary   "
echo "==========================="
cd PreSumm/src
python train.py -task abs -mode test_text -batch_size 100 -test_batch_size 100 -text_src ../raw_data/"${dataset}-seed-${seed}".txt -test_from ../models/model_step_148000.pt -visible_gpus 0 -result_path ../results/"${dataset}-seed-${seed}" -log_file ../logs/"${dataset}-seed-${seed}".log
cd ../..
cp "PreSumm/results/${dataset}-seed-${seed}.-1.candidate" "processed_data/${dataset}/seed-${seed}/train/summary"

echo "==========================="
echo "  begin generate EDA   "
echo "==========================="
cd EDA
python code/augment.py --input="../processed_data/${dataset}/seed-${seed}/train/data" --output="../processed_data/${dataset}/seed-${seed}/train/eda" --num_aug=1
cd ..