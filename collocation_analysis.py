"""Collocation analysis"""

from itertools import chain
from time import strftime
import pickle as p
import os

import nltk
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

from utils import JSTORCorpus

DATA_PATH = 'data/'
CORPUS_PATH = DATA_PATH + 'last-15-years-corpus.p'
OUT_PATH = 'data/collocations-' + strftime('%Y-%m-%d') + '/'
os.makedirs(OUT_PATH)

# import JSTORCorpus
corpus = JSTORCorpus.load(CORPUS_PATH)
corpus.filter_by_type(allowed_types=['journal-article'])

# collocation filters
self_filter = lambda *w: 'self' not in w

# stats
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

# find bigrams with 'self'
print(strftime("%Y-%m-%d - %H:%M : ") + "finding bigrams with 'self' in last 15 years corpus...")
corpus_chain = chain(*corpus.iter_lower())
finder = BigramCollocationFinder.from_words(corpus_chain, window_size=4)
finder.apply_ngram_filter(self_filter)
self_likelihood_bigrams = finder.score_ngrams(bigram_measures.likelihood_ratio)
bigrams_likelihood_path = OUT_PATH + 'scored_self_bigrams_15.p'
with open(bigrams_likelihood_path, mode='wb') as file:
    p.dump(self_likelihood_bigrams, file)
print(strftime("%Y-%m-%d - %H:%M : ") + f"results saved to {bigrams_likelihood_path}")
self_freq_bigrams = finder.score_ngrams(bigram_measures.likelihood_ratio)
bigrams_freq_path = OUT_PATH + 'scored_self_bigrams_15.p'
with open(bigrams_freq_path, mode='wb') as file:
    p.dump(self_freq_bigrams, file)
print(strftime("%Y-%m-%d - %H:%M : ") + f"results saved to {bigrams_freq_path}")

# find trigrams with 'self'
print(strftime("%Y-%m-%d - %H:%M : ") + "finding trigrams with 'self' in last 15 years corpus...")
corpus_chain = chain(*corpus.iter_lower())
finder = TrigramCollocationFinder.from_words(corpus_chain, window_size=10)
finder.apply_ngram_filter(self_filter)
self_likelihood_trigrams = finder.score_ngrams(trigram_measures.likelihood_ratio)
tri_likelihood_path = OUT_PATH + 'whole_corpus_self_likelihood_trigrams.p'
with open(tri_likelihood_path, mode='wb') as file:
    p.dump(self_likelihood_trigrams, file)
print(strftime("%Y-%m-%d - %H:%M : ") + f"results saved to {tri_likelihood_path}")
self_freq_trigrams = finder.score_ngrams(trigram_measures.raw_freq)
tri_freq_path = OUT_PATH + 'whole_corpus_self_freq_trigrams.p'
with open(tri_freq_path, mode='wb') as file:
    p.dump(self_freq_trigrams, file)
print(strftime("%Y-%m-%d - %H:%M : ") + f"results saved to {tri_freq_path}")

print('END')
