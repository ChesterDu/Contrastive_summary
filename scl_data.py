import pickle
import torch
import random
import tqdm
from transformers import RobertaTokenizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer as Summarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from torch.utils.data import TensorDataset, DataLoader

LANGUAGE = "english"

stemmer = Stemmer(LANGUAGE)
summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)

def summarize(x):
    parser = PlaintextParser.from_string(x, Tokenizer(LANGUAGE))
    sentence_count = int(len(parser.document.sentences) / 5.1) + 1
    out = ""
    for sentence in summarizer(parser.document, sentence_count):
        out += str(sentence) + ' '
    return out.strip()



def make_dataset(args):
    if args.dataset == "amazon_2":
        train_dataset, test_dataset = make_amazon_2_dataset(args)
        test_dataset = TensorDataset(test_dataset[:200000][0],test_dataset[:200000][1])
    
    return train_dataset, test_dataset
        

def make_amazon_2_dataset(args):
    if args.dataset_pth != "none":
        return torch.load(args.dataset_pth)
    
    MAX_TRAIN_DATA_NUM_PER_CLASSS = 1800000
    MAX_TEST_DATA_NUM_PER_CLASSS = 200000
    raw_data = pickle.load(open("dataset/amazon_polarity/raw_data.pkl","rb"))
    random.shuffle(raw_data["train"]["pos"])
    random.shuffle(raw_data["train"]["neg"])
    random.shuffle(raw_data["test"]["pos"])
    random.shuffle(raw_data["test"]["neg"])
    max_train_num = int(MAX_TRAIN_DATA_NUM_PER_CLASSS * args.percentage)
    # max_test_num = int(MAX_TEST_DATA_NUM_PER_CLASSS * args.percentage)
    max_test_num = MAX_TEST_DATA_NUM_PER_CLASSS
    train_seqs = {"pos":raw_data["train"]["pos"][:max_train_num], "neg":raw_data["train"]["neg"][:max_train_num]}
    test_seqs = {"pos":raw_data["test"]["pos"][:max_test_num], "neg":raw_data["test"]["neg"][:max_test_num]}

    def seq2id(args,seqs,is_train=True):
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        if is_train:
            max_num = len(seqs["pos"])
            X_pos = torch.LongTensor(max_num,args.max_len)
            X_neg = torch.LongTensor(max_num,args.max_len)
            S_pos = torch.LongTensor(max_num,args.max_len)
            S_neg = torch.LongTensor(max_num,args.max_len)
            S_semi = torch.LongTensor(max_num,args.max_len)
            for i in tqdm.tqdm(range(max_num)):
                x_pos = seqs["pos"][i][0]
                x_neg = seqs["neg"][i][0]
                s_pos = summarize(x_pos)
                s_neg = summarize(x_neg)
                s_semi = s_pos + "\n" + s_neg

                X_pos[i] = tokenizer(x_pos, padding = 'max_length', max_length = args.max_len, truncation = True, return_tensors="pt")["input_ids"]
                X_neg[i] = tokenizer(x_neg, padding = 'max_length', max_length = args.max_len, truncation = True, return_tensors="pt")["input_ids"]
                S_pos[i] = tokenizer(s_pos, padding = 'max_length', max_length = args.max_len, truncation = True, return_tensors="pt")["input_ids"]
                S_neg[i] = tokenizer(s_neg, padding = 'max_length', max_length = args.max_len, truncation = True, return_tensors="pt")["input_ids"]
                S_semi[i] = tokenizer(s_semi, padding = 'max_length', max_length = args.max_len, truncation = True, return_tensors="pt")["input_ids"]
            
            dataset = TensorDataset(X_pos,X_neg,S_pos,S_neg,S_semi)
        
        else:
            max_num = len(seqs["pos"])
            X = torch.LongTensor(max_num*2,args.max_len)
            Y = torch.LongTensor(max_num*2)
            for i in tqdm.tqdm(range(max_num)):
                x_pos = seqs["pos"][i][0]
                x_neg = seqs["neg"][i][0]
                X[i*2] = tokenizer(x_pos, padding = 'max_length', max_length = args.max_len, truncation = True, return_tensors="pt")["input_ids"]
                X[i*2+1] = tokenizer(x_neg, padding = 'max_length', max_length = args.max_len, truncation = True, return_tensors="pt")["input_ids"]
                Y[i*2] = 1
                Y[i*2+1] = 0
            dataset = TensorDataset(X,Y)

        return dataset
    
    train_dataset = seq2id(args,train_seqs,True)
    test_dataset = seq2id(args,test_seqs,False)

    return train_dataset, test_dataset

    
