"""Collocation analysis"""

from itertools import chain
from time import strftime
import pickle as p

import nltk
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

from utils import JSTORCorpus

# import JSTORCorpus
corpus = JSTORCorpus.load('data/whole_corpus.p')

# collocation filters
self_filter = lambda *w: 'self' not in w
rom_filter = lambda *w: 'romantic' not in w
not_rom_filter = lambda *w: 'romantic' in w

# stats
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

# find bigrams with 'self'
print(strftime("%Y-%m-%d - %H:%M : ") + "finding bigrams with 'self'...")
corpus_chain = chain(corpus.iter_lower())
finder = BigramCollocationFinder.from_words(corpus_chain, window_size=4)
finder.apply_ngram_filter(self_filter)
scored_self_bigrams = finder.score_ngrams(bigram_measures.likelihood_ratio)
with open('data/scored_self_bigrams.p', mode='wb') as file:
    p.dump(scored_self_bigrams, file)
print(strftime("%Y-%m-%d - %H:%M : ") + "results saved to data/scored_self_bigrams.p")

# find trigrams with 'self' and 'romantic'
print(strftime("%Y-%m-%d - %H:%M : ") + "finding trigrams with 'self' and 'romantic'...")
corpus_chain = chain(corpus.iter_lower())
finder = TrigramCollocationFinder.from_words(corpus_chain, window_size=10)
finder.apply_ngram_filter(self_filter)
finder.apply_ngram_filter(rom_filter)
scored_self_rom_trigrams = finder.score_ngrams(trigram_measures.likelihood_ratio)
with open('data/scored_self_rom_trigrams.p', mode='wb') as file:
    p.dump(scored_self_rom_trigrams, file)
print(strftime("%Y-%m-%d - %H:%M : ") + "results saved to data/scored_self_rom_trigrams.p")

# find trigrams with 'self' and NOT 'romantic'
print(strftime("%Y-%m-%d - %H:%M : ") + "finding trigrams with 'self' and NOT 'romantic'...")
corpus_chain = chain(corpus.iter_lower())
finder = TrigramCollocationFinder.from_words(corpus_chain, window_size=10)
finder.apply_ngram_filter(self_filter)
finder.apply_ngram_filter(not_rom_filter)
scored_self_not_rom_trigrams = finder.score_ngrams(trigram_measures.likelihood_ratio)
with open('data/scored_self_not_rom_trigrams.p', mode='wb') as file:
    p.dump(scored_self_not_rom_trigrams, file)
print(strftime("%Y-%m-%d - %H:%M : ") + "results saved to data/scored_self_not_rom_trigrams.p")
