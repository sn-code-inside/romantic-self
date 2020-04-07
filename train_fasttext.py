"""Main script for loading corpus and training model"""
from gensim.models import FastText
from utils import JSTORCorpus, load_jstor_corpus

# Import corpus
corpus = load_jstor_corpus('data/last-15-years-corpus.p')

# Initialise and train model
model = FastText(size=200, workers=16)
model.build_vocab(sentences=corpus)
model.train(sentences=corpus, total_examples=len(corpus.corpus_meta), epochs=25)

# Save
model.save('models/last-15-years-25-epochs.ftmodel')
