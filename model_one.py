# coding=utf-8
"""
IBM Model 1 implementation
"""
__author__ = 'keelan'

import numpy
import time
from helper import Alphabet
from reader import parallel_corpus_reader, eval_data_reader
from language_model import BigramModel

class TranslationModel:

    def __init__(self, aligned_sentences, max_iterations=-1, eta=None):
        """

        :param aligned_sentences: a list of tuples of aligned sentences
        :param max_iterations: the number of iterations to run EM
        :param eta: the value that the delta of the EM probabilities must fall below to be considered converged
        """
        self.aligned_sentences = aligned_sentences
        self.e_alphabet = Alphabet()
        self.f_alphabet = Alphabet()
        if eta is None:
            # very simple heuristic
            self.eta = len(aligned_sentences)/100.
        else:
            self.eta = eta
        self.max_iterations = max_iterations
        if max_iterations == -1:
            self.do_more = self.has_converged
        else:
            self.do_more = self.stop_iterations


    def convert_to_vector(self, raw_data, lang="e", training=True):
        """

        :param raw_data: a tokenized sentence
        :param lang: whether it's source or target
        :param training: whether this is during training or testing
        :return: numpy array of the integers corresponding to words
        """
        if lang == "e":
            alphabet = self.e_alphabet
        else:
            alphabet = self.f_alphabet

        if training:
            return numpy.array(map(alphabet.get_index, raw_data), dtype=int)

        else:
            vector = numpy.zeros(len(raw_data), dtype=int)
            for i,word in enumerate(raw_data):
                try:
                    vector[i] = alphabet.get_index(word)
                except KeyError:
                    continue #ignoring OOV words
            return vector

    def populate_alphabets(self):
        """
        Populates the alphabets so that the tokens can have an integer
        representation. Also converts the sentences into this format.

        """
        for e_instance,f_instance in self.aligned_sentences:

            for i,token in enumerate(e_instance.raw_data):
                self.e_alphabet.add(token)

            for i,token in enumerate(f_instance.raw_data):
                self.f_alphabet.add(token)

            e_instance.data = self.convert_to_vector(e_instance.raw_data, "e")
            f_instance.data = self.convert_to_vector(f_instance.raw_data, "f")

    def init_translation_table(self):
        """
        Sets up the class field of the translation table and the cache of
        the previous table in order to do the initial delta.

        Initializes the probability of everything at 0.25

        """
        self.t_table = numpy.zeros([self.e_alphabet.size(), self.f_alphabet.size()])
        self.previous_t_table = numpy.zeros([self.e_alphabet.size(), self.f_alphabet.size()])
        self.t_table.fill(.25)

    def expectation_maximization(self):
        """
        runs the EM algorithm for a specific number of iterations or until
        it has converged.

        """
        i = 0
        while not self.do_more(i):
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
                    tmp = self.t_table[e,f_instance.data]/self.s_total[e]
                    self.counts[e,f_instance.data] += tmp
                    self.total[f_instance.data] += tmp

            # estimate probabilities
            self.t_table = self.counts/self.total

            i += 1

            print "\t{:.3f} seconds".format(time.time()-time1),

    def has_converged(self, i):
        """
        calculates the delta, sees if it is lower than eta

        @param i: only used so this method can have the same signature as stop_iterations
        @return: a boolean whether the EM iterations need to stop
        """
        delta = numpy.sum(numpy.abs(self.t_table - self.previous_t_table))
        self.previous_t_table = numpy.copy(self.t_table)
        if i != 0:
            print "\tdelta: {:.3f}".format(delta)
        if delta < self.eta:
            return True

        return False

    def stop_iterations(self, i):
        """

        @param i: current iteration nubmer
        @return: boolean whether EM need to stop iterating
        """
        print
        return i >= self.max_iterations

    def train(self):
        """
        does all tasks necessary to train our model

        """
        self.populate_alphabets()
        self.init_translation_table()
        self.expectation_maximization()
        self.build_language_model()

    def evaluate(self, candidate_data):
        """
        given candidate translations, this will select the best according to
        our translation table and language model. prints the source sentence
        and best candidate, which is argmax(p(t|s))

        :param candidate_data: a list with a source sentence and translation candidates
        """
        for (source, source_sent_tokenized), candidates in candidate_data:
            candidate_scores = numpy.zeros(len(candidates))
            for i,(c_sent, c) in enumerate(candidates):
                candidate_scores[i] = self.translation_log_prob(c, source_sent_tokenized)
            print u"source sentence: {:s}".format(source)
            print u"best translation: {:s}".format(candidates[numpy.argmax(candidate_scores)][0])
            print

    def t_table_log_prob(self, e_sentence, f_sentence):
        """
        gives the log(p(s|t))

        :param e_sentence: tokenized target sentence
        :param f_sentence: tokenized candidate sentence
        :return: the log probability of a sentence translating to a candidate
        """
        e_sentence = self.convert_to_vector(e_sentence, lang="e", training=False)
        f_sentence = self.convert_to_vector(f_sentence, lang="f", training=False)
        product = 1.
        for e in e_sentence:
            product *= numpy.sum(self.t_table[e,f_sentence])
        return numpy.log(product/(len(f_sentence)**len(e_sentence)))

    def build_language_model(self):
        """
        creates the language model for our target language and saves it in a class
        field

        """
        all_data = []
        for e_sentence,_ in self.aligned_sentences:
            all_data.append(e_sentence.data)
        self.language_model = BigramModel(all_data, self.e_alphabet)
        self.language_model.train()

    def translation_log_prob(self, e_sentence, f_sentence):
        """

        :param e_sentence: tokenized target sentence
        :param f_sentence: tokenized source sentence
        :return: the log(p(s|t)*p())
        """
        return self.t_table_log_prob(e_sentence, f_sentence) + self.language_model.log_prob(e_sentence)

    def save(self):
        raise NotImplementedError

    def load(self, data):
        raise NotImplementedError

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="train or evaluate")
    parser.add_argument("-t", "--target", help="the target language we want to translate to")
    parser.add_argument("-s", "--source", help="the source language we're translating from")
    parser.add_argument("--eval_source", help="source language evaluation data")
    parser.add_argument("--eval_target", help="target language evaluation data")
    parser.add_argument("-i", "--iterations", help="number of iterations to run EM", default=-1, type=int)
    parser.add_argument("--sentences", help="number of sentences to read from aligned data", default=-1, type=int)
    parser.add_argument("-e", "--eta", help="the value which EM's delta must fall below to be considered converged",
                        default=None, type=float)

    args = parser.parse_args()

    data = parallel_corpus_reader(args.source, args.target, max_sents=args.sentences)
    tm = TranslationModel(data, max_iterations=args.iterations, eta=args.eta)
    tm.train()
    if args.mode == "evaluate":
        eval_data = eval_data_reader(args.eval_source, args.eval_target)
        tm.evaluate(eval_data)

