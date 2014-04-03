# coding=utf-8
__author__ = 'keelan'

import numpy
import time
from helper import Alphabet
from reader import parallel_corpus_reader, TOKENIZER
from nltk.model.ngram import NgramModel
from nltk.probability import LidstoneProbDist
from language_model import BigramModel

class TranslationModel:
    ETA = .1

    def __init__(self, aligned_sentences):
        self.aligned_sentences = aligned_sentences
        self.e_alphabet = Alphabet()
        self.f_alphabet = Alphabet()

    def convert_to_vector(self, raw_data, lang="e", training=True):
        if lang == "e":
            alphabet = self.e_alphabet
        else:
            alphabet = self.f_alphabet

        if training:
            return numpy.array(map(alphabet.get_index, raw_data), dtype=int)

        else:
            vector = numpy.zeros(len(raw_data))
            for i,word in enumerate(raw_data):
                try:
                    vector[i] = alphabet.get_index(word)
                except KeyError:
                    continue #ignoring OOV words for now
            return vector

    def populate_alphabets(self):
        for e_instance,f_instance in self.aligned_sentences:

            for i,token in enumerate(e_instance.raw_data):
                self.e_alphabet.add(token)

            for i,token in enumerate(f_instance.raw_data):
                self.f_alphabet.add(token)

            e_instance.data = self.convert_to_vector(e_instance.raw_data, "e")
            f_instance.data = self.convert_to_vector(f_instance.raw_data, "f")

    def init_translation_table(self):
        self.t_table = numpy.zeros([self.e_alphabet.size(), self.f_alphabet.size()])
        self.previous_t_table = numpy.zeros([self.e_alphabet.size(), self.f_alphabet.size()])
        self.t_table.fill(.25)

    def expectation_maximization(self, iterations=20):
        i = 0
        while not self.has_converged():
            time1 = time.time()
            print "iteration {:d}".format(i+1),
            # initialize
            self.total = numpy.zeros(self.f_alphabet.size())
            self.counts = numpy.zeros([self.e_alphabet.size(), self.f_alphabet.size()])

            for e_instance,f_instance in self.aligned_sentences:
                self.s_total = numpy.zeros(self.e_alphabet.size())

                # compute normalization
                for e_word in e_instance.data:
                    self.s_total[e_word] += numpy.sum(self.t_table[e_word, f_instance.data])

                # collect counts
                for e in e_instance.data:
                    #for f in f_instance.data:
                    tmp = self.t_table[e,f_instance.data]/self.s_total[e]
                    self.counts[e,f_instance.data] += tmp
                    self.total[f_instance.data] += tmp

            # estimate probabilities
            self.t_table = self.counts/self.total

            i += 1

            print "\t {:f} seconds".format(time.time()-time1)

    def has_converged(self): # need to fix this
        delta = numpy.sum(numpy.abs(self.t_table - self.previous_t_table))
        self.previous_t_table = numpy.copy(self.t_table)
        print "delta", delta
        if delta < self.ETA:
            return True

        return False

    def train(self):
        self.populate_alphabets()
        self.init_translation_table()
        self.expectation_maximization()
        self.build_language_model()

        """
        for i in range(self.e_alphabet.size()):
            for j in range(self.f_alphabet.size()):
                print self.e_alphabet.get_label(i),
                print self.f_alphabet.get_label(j),
                print self.t_table[i,j]
        """

    def evaluate(self, candidate_data):
        pass

    def t_table_log_prob(self, e_sentence, f_sentence):
        """need to optimize"""
        e_sentence = self.convert_to_vector(e_sentence, lang="e", training=False)
        f_sentence = self.convert_to_vector(f_sentence, lang="f", training=False)
        product = 1.
        for e in e_sentence:
            sent_sum = 0
            for f in f_sentence:
                sent_sum += self.t_table[e,f]
            product *= sent_sum
        return numpy.log(product/(len(f_sentence)**len(e_sentence)))

    def build_language_model(self):
        all_data = []
        for e_sentence,_ in self.aligned_sentences:
            all_data.append(e_sentence.data)
        self.language_model = BigramModel(all_data, self.e_alphabet)
        self.language_model.train()

    def translation_log_prob(self, e_sentence, f_sentence):
        e_sentence = TOKENIZER.tokenize(e_sentence)
        f_sentence = TOKENIZER.tokenize(f_sentence)
        return self.t_table_log_prob(e_sentence, f_sentence) + self.language_model.log_prob(e_sentence)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--english", help="aligned English data")
    parser.add_argument("-f", "--foreign", help="aligned 'foreign' data")
    parser.add_argument("-i", "--iterations", help="number of iterations to run")

    args = parser.parse_args()

    data = parallel_corpus_reader(args.english, args.foreign, max_sents=100)
    tm = TranslationModel(data)
    tm.train()

    print tm.translation_log_prob("the book", "el libro")
    print tm.translation_log_prob("book the", "el libro")
    print tm.translation_log_prob("you are quite right", "tiene toda la razón del mundo")
    print tm.translation_log_prob("you are quite wrong", "tiene toda la razón del mundo")
