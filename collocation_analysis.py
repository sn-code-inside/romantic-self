"""Collocation analysis"""

from itertools import chain
from time import strftime
import pickle as p
from copy import deepcopy

import nltk
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

from utils import JSTORCorpus

DATA_PATH = 'data/'
CORPUS_PATH = DATA_PATH + 'whole-corpus.p'

# import JSTORCorpus
corpus = JSTORCorpus.load(CORPUS_PATH)

# collocation filters
self_filter = lambda *w: 'self' not in w
rom_filter = lambda *w: 'romantic' not in w

# stats
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

# find bigrams with 'self'
print(strftime("%Y-%m-%d - %H:%M : ") + "finding bigrams with 'self' in whole corpus...")
corpus_chain = chain(*corpus.iter_lower())
finder = BigramCollocationFinder.from_words(corpus_chain, window_size=4)
finder.apply_ngram_filter(self_filter)
scored_self_bigrams = finder.score_ngrams(bigram_measures.likelihood_ratio)
bigrams_path = DATA_PATH + 'scored_self_bigrams.p'
with open(bigrams_path, mode='wb') as file:
    p.dump(scored_self_bigrams, file)
print(strftime("%Y-%m-%d - %H:%M : ") + f"results saved to {bigrams_path}")

# find trigrams with 'self'
print(strftime("%Y-%m-%d - %H:%M : ") + "finding trigrams with 'self' in whole corpus...")
corpus_chain = chain(*corpus.iter_lower())
finder = TrigramCollocationFinder.from_words(corpus_chain, window_size=10)
finder.apply_ngram_filter(self_filter)
self_likelihood_trigrams = finder.score_ngrams(trigram_measures.likelihood_ratio)
tri_likelihood_path = DATA_PATH + 'whole_corpus_self_likelihood_trigrams.p'
with open(tri_likelihood_path, mode='wb') as file:
    p.dump(self_likelihood_trigrams, file)
print(strftime("%Y-%m-%d - %H:%M : ") + f"results saved to {tri_likelihood_path}")
self_freq_trigrams = finder.score_ngrams(trigram_measures.raw_freq)
tri_freq_path = DATA_PATH + 'whole_corpus_self_freq_trigrams.p'
with open(tri_freq_path, mode='wb') as file:
    p.dump(self_freq_trigrams, file)
print(strftime("%Y-%m-%d - %H:%M : ") + f"results saved to {tri_freq_path}")

# 10-year rolling windows
print(strftime("%Y-%m-%d - %H:%M : ") + "trigrams from 10-year rolling windows")
for lower in range(1950, 2015, 5):
    upper = lower + 10
    subset = deepcopy(corpus)
    subset.filter_by_year(lower, upper)
    print(strftime("%Y-%m-%d - %H:%M : ") + f"{lower} : {upper} : n = {len(subset)}")
    subset_chain = chain(*subset.iter_lower())
    finder = TrigramCollocationFinder.from_words(subset_chain, window_size=10)
    finder.apply_ngram_filter(self_filter)
    window_likelihood_trigrams = finder.score_ngrams(trigram_measures.likelihood_ratio)
    window_likelihood_path = DATA_PATH + f'windows/{lower}_{upper}_likelihood.p'
    with open(window_likelihood_path, mode='wb') as file:
        p.dump(window_likelihood_trigrams, file)
    print(strftime("%Y-%m-%d - %H:%M : ") + f"results saved to {window_likelihood_path}")
    window_freq_trigrams = finder.score_ngrams(trigram_measures.raw_freq)
    window_freq_path = DATA_PATH + f'windows/{lower}_{upper}_freq.p'
    with open(window_freq_path, mode='wb') as file:
        p.dump(window_freq_trigrams, file)
    print(strftime("%Y-%m-%d - %H:%M : ") + f"results saved to {window_freq_path}")

print('END')
