#!/bin/bash
seed=$1

echo "Runing Experiment-Seed-${seed}"
mkdir "Experiment"
mkdir "Experiment/Seed-${seed}"

# for dataset in "amazon" "yelp" "ag_news";
# for dataset in "yahoo";
# do
#   sh scripts/process_data.sh $dataset $seed
# done
# sh scripts/process_data.sh yelp $seed
# sh scripts/process_data.sh amazon $seed
# sh scripts/process_data.sh ag_news $seed

cd src
# for dataset in "amazon" "yelp" "ag_news";
for dataset in "imdb";
do
for model in "bert" "xlnet" "roberta";
# for model in "bert" "xlnet";
# for model in "xlnet";
do
mkdir "../Experiment/Seed-${seed}/${model}"
mkdir "../Experiment/Seed-${seed}/${model}/${dataset}"
python main.py --steps 800 --dataset "${dataset}" --model "${model}" --log_step 10 --seed $seed --eval_batch_size 64 --lambd 1
python main.py --steps 800 --dataset "${dataset}" --model "${model}" --log_step 10 --seed $seed --eval_batch_size 64 --lambd 0.9
python main.py --steps 800 --dataset "${dataset}" --model "${model}" --log_step 10 --seed $seed --eval_batch_size 64 --lambd 0.9 --with_summary --aug_methods eda
python main.py --steps 800 --dataset "${dataset}" --model "${model}" --log_step 10 --seed $seed --eval_batch_size 64 --lambd 0.9 --with_summary --aug_methods summary
python main.py --steps 800 --dataset "${dataset}" --model "${model}" --log_step 10 --seed $seed --eval_batch_size 64 --lambd 0.9 --with_summary --aug_methods summary+eda
python main.py --steps 800 --dataset "${dataset}" --model "${model}" --log_step 10 --seed $seed --eval_batch_size 64 --lambd 0.9 --with_summary --aug_methods summary --with_mix
python main.py --steps 800 --dataset "${dataset}" --model "${model}" --log_step 10 --seed $seed --eval_batch_size 64 --lambd 0.9 --with_summary --aug_methods summary+eda --with_mix
done
done
