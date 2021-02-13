from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer as Summarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

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
