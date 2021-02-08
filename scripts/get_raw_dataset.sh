#!/bin/bash

amazon_url="https://s3.amazonaws.com/fast-ai-nlp/amazon_review_full_csv.tgz"
yelp_url="https://s3.amazonaws.com/fast-ai-nlp/yelp_review_full_csv.tgz"
ag_news_url="https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz"
dataset_dir="raw_datasets"

if [ ! -d "${dataset_dir}" ]; then
    mkdir -p "${dataset_dir}"
fi

if test $1 = "amazon" 
then
    url=${amazon_url}
fi

if test $1 = "yelp" 
then
    url=${yelp_url}
fi

if test $1 = "ag_news" 
then
    url=${ag_news_url}
fi

curl "${url}" -o "${dataset_dir}/$1.tgz"
tar -xvf "${dataset_dir}/$1.tgz" -C "${dataset_dir}"
rm "${dataset_dir}/$1.tgz"