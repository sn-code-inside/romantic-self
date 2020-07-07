"""Collocation analysis"""

from time import strftime
import pickle as p
import os
import argparse

from utils import JSTORCorpus, TargetedTrigramAssocFinder

parser = argparse.ArgumentParser(description="Perform collocation analysis on corpus")
parser.add_argument('-w', dest='window', type=int, help='the window size')
args = parser.parse_args()

DATA_PATH = 'data/'
CORPUS_FILE = 'last-15-years-corpus.p'
WINDOW_SIZE = args.window
OUT_PATH = 'data/associations-' + strftime('%Y-%m-%d') + f'-wn{WINDOW_SIZE}/'

# create output directory
os.makedirs(OUT_PATH)

# construct JSTORCorpus, filter, save
if CORPUS_FILE in os.listdir(DATA_PATH):
    corpus = JSTORCorpus.load(DATA_PATH + CORPUS_FILE)
else:
    corpus = JSTORCorpus(DATA_PATH + 'metadata', DATA_PATH + 'ocr')
    corpus.filter_by_year(min_year=2001, max_year=2015)
    corpus.save(DATA_PATH + CORPUS_FILE)

# find trigrams with both target words
print(
    strftime("%Y-%m-%d - %H:%M : ") +
    f"finding trigrams with 'romantic' and 'self' in {DATA_PATH + CORPUS_FILE} " +
    f"with a window_size of {WINDOW_SIZE}..."
    )
finder = TargetedTrigramAssocFinder.from_corpus(corpus.iter_lower(), ('romantic', 'self'), WINDOW_SIZE)
TRI_OUT = OUT_PATH + 'romantic-self-trigrams.p'
with open(TRI_OUT, 'wb') as file:
    p.dump(finder, file)
print(strftime("%Y-%m-%d - %H:%M : ") + f"results saved to {TRI_OUT}")

# extract 'self' bigrams
self_bigrams = finder.bigram_finder()
self_filter = lambda *w: 'self' not in w
self_bigrams.apply_ngram_filter(self_filter)
if ('self','romantic') in self_bigrams.ngram_fd:
    self_bigrams.ngram_fd[('romantic','self')] = self_bigrams.ngram_fd[('self','romantic')]
    del self_bigrams.ngram_fd[('self','romantic')]
SELF_OUT = OUT_PATH + 'self-bigrams.p'
with open(SELF_OUT, 'wb') as file:
    p.dump(self_bigrams, file)
print(strftime("%Y-%m-%d - %H:%M : ") + f"'self' bigrams saved to {SELF_OUT}")

# extract 'romantic' bigrams
romantic_bigrams = finder.bigram_finder()
romantic_filter = lambda *w: 'romantic' not in w
romantic_bigrams.apply_ngram_filter(romantic_filter)
if ('romantic','self') in self_bigrams.ngram_fd:
    self_bigrams.ngram_fd[('self','romantic')] = self_bigrams.ngram_fd[('romantic','self')]
    del self_bigrams.ngram_fd[('romantic','self')]
ROM_OUT = OUT_PATH + 'romantic-bigrams.p'
with open(ROM_OUT, 'wb') as file:
    p.dump(romantic_bigrams, file)
print(strftime("%Y-%m-%d - %H:%M : ") + f"'romantic' bigrams saved to {ROM_OUT}")

print('END')
