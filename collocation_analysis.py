"""Collocation analysis"""

from time import strftime
import pickle as p
import os

import nltk

from utils import JSTORCorpus, TargetedCollocationFinder
# from utils import CorpusBigramCollocationFinder

DATA_PATH = 'data/'
CORPUS_PATH = DATA_PATH + 'last-15-years-corpus.p'
WINDOW_SIZE = 10
OUT_PATH = 'data/collocations-' + strftime('%Y-%m-%d') + f'-wn{WINDOW_SIZE}/'

# create output directory
os.makedirs(OUT_PATH)

# import JSTORCorpus
corpus = JSTORCorpus.load(CORPUS_PATH)

# stats
bigram_measures = nltk.collocations.BigramAssocMeasures()

# # ALL BIGRAMS
# # find bigrams with 'self'
# print(strftime("%Y-%m-%d - %H:%M : ") + "finding all bigrams with 'self' in last 15 years...")

# # iterate over corpus and construct finder from tokens
# finder = CorpusBigramCollocationFinder.from_corpus(corpus.iter_lower(), WINDOW_SIZE)
# # filter out bigrams without 'self'
# self_filter = lambda *w: 'self' not in w
# finder.apply_ngram_filter(self_filter)
# # save
# ALL_BIGRAMS_PTH = OUT_PATH + 'all-bigrams-finder.p'
# with open(ALL_BIGRAMS_PTH, mode='wb') as file:
#     p.dump(finder, file)
# print(strftime("%Y-%m-%d - %H:%M : ") + f"results saved to {ALL_BIGRAMS_PTH}")

# CONTEXTUALISED BIGRAMS
# find collocations in the context of 'romantic'
print(strftime("%Y-%m-%d - %H:%M : ") + "finding bigrams with 'self' with 'romantic'...")
finder = TargetedCollocationFinder.from_corpus(
    corpus.iter_lower(), 'self', include=['romantic'], window_size=WINDOW_SIZE)
# save
ROM_BIGRAMS_PTH = OUT_PATH + 'rom-bigrams-finder.p'
with open(ROM_BIGRAMS_PTH, 'wb') as file:
    p.dump(finder, file)
print(strftime("%Y-%m-%d - %H:%M : ") + f"results saved to {ROM_BIGRAMS_PTH}")

# find collocations outside that context
print(strftime("%Y-%m-%d - %H:%M : ") + "finding bigrams with 'self' without 'romantic'...")
finder = TargetedCollocationFinder.from_corpus(
    corpus.iter_lower(), 'self', exclude=['romantic'], window_size=WINDOW_SIZE)
# save
NONROM_BIGRAMS_PTH = OUT_PATH + 'nonrom-bigrams-finder.p'
with open(NONROM_BIGRAMS_PTH, 'wb') as file:
    p.dump(finder, file)
print(strftime("%Y-%m-%d - %H:%M : ") + f"results saved to {NONROM_BIGRAMS_PTH}")

print('END')
