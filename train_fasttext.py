# Main script for loading corpus and training model
from gensim.models import FastText
from utils import JSTORCorpus

# Import corpus
corpus = JSTORCorpus('data/metadata', 'data/ocr')
corpus.filter_by_year(min_year=2000, max_year=2015)
corpus.save('data/last-15-years-corpus.p')

# Initialise and train model
model = FastText(size=200, workers=16)
model.build_vocab(sentences=corpus)
model.train(sentences=corpus, total_examples=len(corpus), epochs=10)

# Save
model.save('models/last-15-years.ftmodel')