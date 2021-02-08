from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer as Summarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANGUAGE = "english"
SENTENCES_COUNT = 1



class SummarizerWrap():
    def __init__(self,method):
        self.method = method
        self.stemmer = Stemmer(LANGUAGE)
        self.summarizer = Summarizer(self.stemmer)
        self.summarizer.stop_words = get_stop_words(LANGUAGE)

    def sum_text(self, x):
        parser = PlaintextParser.from_string(x, Tokenizer(LANGUAGE))
        out = []
        for sentence in self.summarizer(parser.document, SENTENCES_COUNT):
            out.append(sentence)
            
        return str(out[0])
