# Contrastive Summarization
<!-- **This code is for the paper** "Constructing contrastive samples via summarization for text classification with limited annotations" -->

## Requirements
```
torch==1.7.0
transformers==4.2.2
nltk==3.2.5
summy==0.8.1
sentencepiece==0.1.95
multiprocess==0.70.9
pyrouge==0.1.3
pytorch-transformers==1.2.0
tensorboardX==1.9
```

## PreSumm
We use [PreSumm(Liu and Lapata)](https://arxiv.org/abs/1908.08345) to generate abstractive summary. The summary generation code and pretrained models can be checked out [here](https://github.com/nlpyang/PreSumm/tree/master).
Clone the **PreSumm repo** by:
```
git clone https://github.com/nlpyang/PreSumm.git PreSumm
```
Then **switch to the dev branch in PreSumm repo** and download pretrained models(Liu and Lapata) from [google drive](https://drive.google.com/file/d/1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr/view). Unzip the downloaed file and **move `.pt` file to `./PreSumm/models`**

Then apply the provided patch file `torch170.patch` to adapt the original PreSumm code to `torch==1.7.0` environment.
```
cd PreSumm/src
cat ../../torch170.patch | patch -p1
```


## Dataset
### step1: Download raw dataset
You can choose `$dataset` among  `[amazon,yelp,ag_news]`
```
sh scripts/get_raw_dataset.sh $dataset
```
The raw dataset will be downloaded to `./raw_datasets/$dataset`
### step2: Process Data
You can choose `$dataset` among  `[amazon,yelp,ag_news]`. `$seed` is the random seed for sampling data.
```
sh scripts/process_data.sh $dataset $seed
```
This step will sample the train data and test data, also generate summary of the train data using PreSumm. The processed data is under `./processed_data/$dataset`.
### step3: Run the Experiment


