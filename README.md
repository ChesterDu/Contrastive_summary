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
We use [PreSumm(Liu and Lapata)](https://arxiv.org/abs/1908.08345) to generate abstractive summary. The summary generation code and pretrained models can be checked out [here](https://github.com/nlpyang/PreSumm/tree/master).

Clone the repo by:
```
git clone https://github.com/nlpyang/PreSumm.git PreSumm
```
Then **switch to the dev branch** and download pretrained models(Liu and Lapata) from [google drive](https://drive.google.com/file/d/1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr/view). Unzip the downloaed file and **move `.pt` file to `PreSumm/models`**

