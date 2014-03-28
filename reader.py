__author__ = 'keelan'

import codecs
from nltk.tokenize import WordPunctTokenizer
from helper import Instance

TOKENIZER = WordPunctTokenizer()

def parallel_corpus_reader(f1, f2, max_sents=100):
    total_seen = 0
    result = []
    with codecs.open(f1, "r", "utf-8") as ef_in, codecs.open(f2, "r", "utf-8") as ff_in:
        while total_seen <= max_sents:
            result.append(map(prepare_sentence, [ef_in.readline(), ff_in.readline()]))
            total_seen += 1
    return result

def prepare_sentence(sent):
    toks = [s.lower() for s in TOKENIZER.tokenize(sent)]
    return Instance(raw_data=toks)