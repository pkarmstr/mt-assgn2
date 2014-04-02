__author__ = 'keelan'

import numpy
from helper import Alphabet
from reader import parallel_corpus_reader
from nltk.model.ngram import NgramModel
from nltk.probability import LidstoneProbDist

class TranslationModel:

    def __init__(self, aligned_sentences):
        self.aligned_sentences = aligned_sentences
        self.e_alphabet = Alphabet()
        self.f_alphabet = Alphabet()

    def convert_to_sparse_vector(self, raw_data, lang="e"):
        if lang == "e":
            return numpy.array(map(self.e_alphabet.get_index, raw_data), dtype=int)

        return numpy.array(map(self.f_alphabet.get_index, raw_data), dtype=int)

    def populate_alphabets(self):
        for e_instance,f_instance in self.aligned_sentences:

            for i,token in enumerate(e_instance.raw_data):
                self.e_alphabet.add(token)

            for i,token in enumerate(f_instance.raw_data):
                self.f_alphabet.add(token)

            e_instance.data = self.convert_to_sparse_vector(e_instance.raw_data, "e")
            f_instance.data = self.convert_to_sparse_vector(f_instance.raw_data, "f")

    def init_translation_table(self):
        self.t_table = numpy.zeros([self.e_alphabet.size(), self.f_alphabet.size()])
        self.t_table.fill(.25)

    def expectation_maximization(self, iterations=20):
        for i in range(iterations):
            print "iteration {:d}".format(i+1)
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
                    for f in f_instance.data:
                        self.counts[e,f] += self.t_table[e,f]/self.s_total[e]
                        self.total[f] += self.t_table[e,f]/self.s_total[e]

            # estimate probabilities
            for f in range(self.f_alphabet.size()):
                for e in range(self.e_alphabet.size()):
                    self.t_table[e,f] = self.counts[e,f]/self.total[f]



    def train(self):
        self.populate_alphabets()
        self.init_translation_table()
        self.expectation_maximization()

        """
        for i in range(self.e_alphabet.size()):
            for j in range(self.f_alphabet.size()):
                print self.e_alphabet.get_label(i),
                print self.f_alphabet.get_label(j),
                print self.t_table[i,j]
        """

    def evaluate(self, candidate_data):
        pass

    def sentence_translation_probability(self, e_sentence, f_sentence):
        """need to optimize"""
        e_sentence = self.convert_to_sparse_vector(e_sentence, lang="e")
        f_sentence = self.convert_to_sparse_vector(f_sentence, lang="f")
        product = 1.
        for e in e_sentence:
            sent_sum = 0
            for f in f_sentence:
                sent_sum += self.t_table[e,f]
            product *= sent_sum
        return product/(len(f_sentence)**len(e_sentence))

    def build_language_model(self):
        all_text = [word for e_instance,f_instance in self.aligned_sentences \
                    for word in e_instance.raw_data[:-1]]
        est = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
        self.lm = NgramModel(3, all_text, estimator=est)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--english", help="aligned English data")
    parser.add_argument("-f", "--foreign", help="aligned 'foreign' data")
    parser.add_argument("-i", "--iterations", help="number of iterations to run")

    args = parser.parse_args()

    data = parallel_corpus_reader(args.english, args.foreign, max_sents=1000)
    tm = TranslationModel(data)
    tm.train()
    print tm.sentence_translation_probability("the book".split(), "das buch".split())
