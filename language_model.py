__author__ = 'keelan'

import numpy
import sys

class BigramModel:

    LOG_EPSILON = numpy.log(numpy.finfo(float).eps)

    def __init__(self, data, word_alphabet):
        self.data = data
        self.word_alphabet = word_alphabet
        self.word_alphabet.add("__START__")
        self.start_id = self.word_alphabet.get_index("__START__")
        self.bigram_counts = numpy.zeros([self.word_alphabet.size(), self.word_alphabet.size()])

    def build_counts(self):
        for sentence in self.data:
            self.bigram_counts[self.start_id, sentence[0]] += 1
            self.bigram_counts[sentence[:-1], numpy.roll(sentence, -1)[:-1]] += 1

        # add-one smoothing
        self.bigram_counts += 1

    def build_probabilities(self):
        self.bigram_probabilities = self.bigram_counts/numpy.sum(self.bigram_counts, axis=0)

    def train(self):
        self.build_counts()
        self.build_probabilities()

    def convert_to_vector(self, tokenized_sentence):
        vector = numpy.zeros(len(tokenized_sentence)+1, dtype=int)
        vector[0] = self.start_id
        for i,word in enumerate(tokenized_sentence):
            try:
                vector[i+1] = self.word_alphabet.get_index(word)
            except KeyError:
                vector[i+1] = -1
        return vector

    def log_prob(self, tokenized_sentence):
        lp = 0.
        vector = self.convert_to_vector(tokenized_sentence)
        for i in xrange(len(vector)-1):
            j = i+1
            if vector[i] == -1 or vector[j] == -1:
                lp += self.LOG_EPSILON
            else:
                lp += numpy.log(self.bigram_probabilities[vector[i],vector[j]])
        return lp