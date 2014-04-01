__author__ = 'keelan'

import codecs
from collections import defaultdict,namedtuple
from nltk.tokenize import WordPunctTokenizer
from helper import Instance

TOKENIZER = WordPunctTokenizer()
CandidateTranslations = namedtuple("CandidateTranslations", ["source_sent", "candidates"])

def parallel_corpus_reader(f1, f2, max_sents=100):
    total_seen = 0
    result = []
    with codecs.open(f1, "r", "utf-8") as ef_in, codecs.open(f2, "r", "utf-8") as ff_in:
        for line in ef_in:
            result.append(map(prepare_sentence, [line, ff_in.readline()]))
            total_seen += 1
            if total_seen >= max_sents:
                break
    return result

def eval_data_reader(source, target):
    candidate_translations = defaultdict(list)
    source_sentences = {}
    all_data = []
    with codecs.open(source, "r", "utf-8") as f_in:
        for line in f_in:
            id_,sent = line.rstrip().split("\t")
            source_sentences[id_] = TOKENIZER.tokenize(sent)

    with codecs.open(target, "r", "utf-8") as f_in:
        for line in f_in:
            id_,sent = line.rstrip().split("\t")
            candidate_translations[id_].append(TOKENIZER.tokenize(sent))

    for id_,source_sent in source_sentences.iteritems():
        candidates = candidate_translations[id_]
        all_data.append(CandidateTranslations(source_sent, candidates))

    return all_data



def prepare_sentence(sent):
    toks = [s.lower() for s in TOKENIZER.tokenize(sent)]
    toks.append("__NULL__")
    return Instance(raw_data=toks)