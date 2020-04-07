from utils import JSTORCorpus
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel

# Import and save whole corpus
corpus = JSTORCorpus(data_dir='data/ocr', meta_dir='data/metadata')
corpus.save('data/whole-corpus.p')

# Create dictionary
dct = Dictionary(corpus.iter_lower())
dct.filter_extremes(no_below=5)
dct.filter_n_most_frequent(remove_n=50)
dct.save('models/corpus-lower-dct')

# Train tf-idf model
bow_corpus = (dct.doc2bow(text) for text in corpus.iter_lower())
tfidf_model = TfidfModel(bow_corpus)
tfidf_model.save('models/corpus-lower-tfidf')

# Train lda model
bow_corpus = (dct.doc2bow(text) for text in corpus.iter_lower())
lda_model = LdaModel(bow_corpus, num_topics=150)
lda_model.save('models/corpus-lower-lda')

