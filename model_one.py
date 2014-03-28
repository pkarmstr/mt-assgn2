__author__ = 'keelan'

import numpy
from helper import Alphabet
from reader import parallel_corpus_reader

class TranslationModel:

    def __init__(self, aligned_sentences):
        self.aligned_sentences = aligned_sentences
        self.e_alphabet = Alphabet()
        self.f_alphabet = Alphabet()

    def convert_to_sparse_vector(self, raw_data, lang="e"):
        if lang == "e":
            return numpy.array(map(self.e_alphabet.get_index, raw_data))

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
        self.t_table = numpy.ones([self.e_alphabet.size(), self.f_alphabet.size()])

    def expectation_maximization(self, iterations=10):
        for i in range(iterations):
            print "iteration {:d}".format(i)
            # initialize
            self.counts = numpy.zeros([self.e_alphabet.size(), self.f_alphabet.size()])
            self.s_total = numpy.zeros(self.e_alphabet.size())
            self.total = numpy.zeros(self.f_alphabet.size())

            for e_instance,f_instance in self.aligned_sentences:
                # compute normalization
                for e_word in e_instance.data:
                    self.s_total[e_word] = numpy.sum(self.t_table[e_word, f_instance.data])

                # collect counts
                for e in e_instance.data:
                    for f in f_instance.data:
                        self.counts[e,f] += self.t_table[e,f]/self.s_total[e]
                        self.total[f] += self.t_table[e,f]/self.s_total[e]

            # estimate probabilities
            for f in f_instance.data:
                for e in e_instance.data:
                    self.t_table[e,f] = self.counts[e,f]/self.total[f]


    def train(self):
        self.populate_alphabets()
        self.init_translation_table()
        self.expectation_maximization()

        for i in range(10):
            for j in range(10):
                print self.e_alphabet.get_label(i),
                print self.f_alphabet.get_label(j),
                print self.t_table[i,j]

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