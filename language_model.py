__author__ = 'keelan'

import numpy

class BigramModel:

    def __init__(self, data, word_alphabet):
        self.data = data
        self.word_alphabet = word_alphabet
        self.word_alphabet.add("__START__")
        self.start_id = self.word_alphabet.get_index("__START__")
        self.unigram_counts = numpy.zeros(self.word_alphabet.size())
        self.bigram_counts = numpy.zeros([self.word_alphabet.size(), self.word_alphabet.size()])

    def build_counts(self):
        for sentence in self.data:
            self.unigram_counts[sentence] += 1
            self.bigram_counts[self.start_id, sentence[0]] += 1
            self.bigram_counts[sentence[:-1], numpy.roll(sentence, -1)[:-1]] += 1

    def build_probabilities(self):
        self.unigram_probabilities = self.unigram_counts/numpy.sum(self.unigram_counts)
        self.bigram_probabilities = self.bigram_probabilities/numpy.sum(self.bigram_counts, axis=0)

    def log_prob(self):
        pass