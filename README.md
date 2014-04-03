IBM Model One, implemented for Spanish to English

This module is written for python2.7!

Dependencies not found in the standard lib include: NLTK and numpy

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  train or evaluate
  -t TARGET, --target TARGET
                        the target language we want to translate to
  -s SOURCE, --source SOURCE
                        the source language we're translating from
  --eval_source EVAL_SOURCE
                        source language evaluation data
  --eval_target EVAL_TARGET
                        target language evaluation data
  -i ITERATIONS, --iterations ITERATIONS
                        number of iterations to run EM
  --sentences SENTENCES
                        number of sentences to read from aligned data
  -e ETA, --eta ETA     the value which EM's delta must fall below to be
                        considered converged

typical usage would be something like:

python model_one.py -m evaluate -s corpus.es -t corpus.en --eval_source eval.es --eval_target eval.en --sentences 1000 -i 25

If one leaves out the -i flag, then EM will run until it converges. You can also adjust the value used to
determine convergence with the -e flag.

As of right now, functionality to save and load translation tables has not yet been implemented.

If you want to evaluate how well your translation model works, in the test-resources folder, eval.en and eval.es can be
useful.  The first candidate sentence should be the one returned during an evaluation.