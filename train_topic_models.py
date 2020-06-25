"""Basic script for training Lda and Tfidf models"""

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel
from nltk.corpus import stopwords

from utils import JSTORCorpus

# Import and save whole corpus
corpus = JSTORCorpus(data_dir='data/ocr', meta_dir='data/metadata')
corpus.save('data/whole-corpus.p')

# Create dictionary
dct = Dictionary(corpus.iter_lower())
print(f"Dictionary created: {len(dct.token2id)} unique types")
# Only keep types that appear in at least 20 documents
dct.filter_extremes(no_below=20, no_above=1., keep_n=None)
print(f"Infrequent types filtered out: {len(dct.token2id)} unique types")
# Get list of English stopwords
sw_tokens = stopwords.words('english')
# Convert to ids for dct
sw_ids = [dct.token2id[token] for token in sw_tokens if token in dct.token2id]
# Remove stopwords from dct
dct.filter_tokens(bad_ids=sw_ids)
print(f"Stopwords filtered out: {len(dct.token2id)} unique types")
# Shrink so there are no empty dimensions
dct.compactify()
print("Dictionary compactified")
dct.save('models/corpus-lower-dct')
print("Dictionary saved to models/corpus-lower-dct")

# Train tf-idf model
print("Training TfidfModel...")
bow_corpus = (dct.doc2bow(text) for text in corpus.iter_lower())
tfidf_model = TfidfModel(bow_corpus)
tfidf_model.save('models/corpus-lower-tfidf')
print("TfidfModel trained and saved to models/corpus-lower-tfidf")

# Train lda model
print("Training LdaModel...")
bow_corpus = (dct.doc2bow(text) for text in corpus.iter_lower())
lda_model = LdaModel(bow_corpus, num_topics=150)
lda_model.save('models/corpus-lower-lda')
print("LdaModel trained and saved to models/corpus-lower-tfidf")
