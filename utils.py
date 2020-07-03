"""Utilities for accessing JSTOR Data for Research Corpus"""

import os
import re
import time
import pickle as p
from math import inf

from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams

class JSTORCorpus(object):
    """Iterator for streaming files into Gensim. Also allows basic filtering.

    Arguments:
    - meta_dir (str): path to xml files
    - data_dir (str): path to txt files
    - corpus_meta (dict, optional): corpus dict, used internally"""

    # For cleaning txt files. Finds xml tags or end-of-line hyphens to delete
    CLEAN_RGX = re.compile(r'<[^>]+>|(?<=\w)-\s+(?=\w)')

    def __init__(self, meta_dir, data_dir, corpus_meta=None):
        self.meta_dir = meta_dir
        self.data_dir = data_dir
        self.corpus_meta = corpus_meta
        self.doc_types = set()

        # Ingest corpus if no existing corpus provided
        if self.corpus_meta is None:
            self.extract_jstor_meta(self.meta_dir, self.data_dir)
        else:
            # Otherwise loop over the corpus and extract doc_type information
            self.doc_types = set([doc['type'] for key, doc in self.corpus_meta.items()])

    def __iter__(self):
        for key in self.corpus_meta:
            with open(key) as file:
                # Get text
                raw_xml = file.read()
                # Strip tags
                text = self.CLEAN_RGX.sub('', raw_xml)
                # Yield array of tokens
                yield wordpunct_tokenize(text)

    def __len__(self):
        return len(self.corpus_meta)

    def iter_lower(self):
        """Iterates over the corpus, putting tokens in lower case."""

        for key in self.corpus_meta:
            with open(key) as file:
                # Get text
                raw_xml = file.read()
                # Strip tags
                text = self.CLEAN_RGX.sub('', raw_xml)
                # Yield array of lowercase tokens
                yield wordpunct_tokenize(text.lower())

    def extract_jstor_meta(self, meta_dir, data_dir):
        """Loops over directory of JSTOR metadata files, extracts key info from xml

        Arguments:
        - meta_dir (str): directory where metadata files are held
        - data_dir (str): directory where data files are held
        """

        self.corpus_meta = {}

        parsed = 0
        skipped = 0

        print(f'Parsing xml files in {meta_dir}. Associated .txt in {data_dir}')

        # The metadata file contains many documents without a text file. We don't want that!
        actual_docs = set(os.listdir(data_dir))

        for name in os.listdir(meta_dir):

            # Infer name of data file and check
            txt_file = name[:-3] + 'txt' # replace .xml with .txt
            if txt_file not in actual_docs:
                skipped += 1
                continue

            # Get doi (for book metadata)
            doi = re.sub('^.+_', '', name[:-4])

            # Locate data file
            data_file = os.path.join(data_dir, txt_file) # fill path

            # Read in metadata file
            with open(os.path.join(meta_dir, name)) as file:
                meta_xml = BeautifulSoup(file.read(), features="lxml")

            # Get key metadata
            doc_dict = {}

            # For articles:
            if name.startswith('journal-article'):
                doc_dict['type'] = meta_xml.html.body.article['article-type']
                # Store doc type in corpus metadata
                self.doc_types.add(doc_dict['type'])
                title = meta_xml.find(['article-title', 'source'])
                if title is not None:
                    doc_dict['title'] = title.get_text()
                year = meta_xml.find('year')
                if year is not None:
                    doc_dict['year'] = year.get_text()

            # For book chapters:
            elif name.startswith('book-chapter'):
                doc_dict['type'] = 'book-chapter'
                self.doc_types.add('book-chapter')
                # First book-id element is id of whole book
                part_of = meta_xml.find('book-id')
                if part_of is not None:
                    doc_dict['part-of'] = part_of.get_text()
                year = meta_xml.find('year')
                if year is not None:
                    doc_dict['year'] = year.get_text()
                # Getting chapter title is slightly harder, because sometimes each book-part
                # is labelled simply with the internal id, and sometimes with the doi
                book_id = re.sub('.+_', '', doi)
                book_rgx = re.compile(re.escape(book_id))
                doc_dict['title'] = meta_xml.find(
                    'book-part-id', string=book_rgx).parent.find('title').get_text()

            # Store in corpus_meta dict
            self.corpus_meta[data_file] = doc_dict

            # Increment counter
            parsed += 1

        # Success message
        print(f'{parsed} documents parsed successfully. {skipped} documents skipped.')

    def filter_by_year(self, min_year=1750, max_year=inf):
        """Filters the corpus according to minimum and maximum years

        Arguments:
        - min_year (int)
        - max_year (int)"""

        filtered_corpus = {}

        orig_len = len(self)
        print(f'Filtering {orig_len} documents between years {min_year} and {max_year}...')

        for key, val_dict in self.corpus_meta.items():
            # Skip files that cannot be parsed
            if 'year' not in val_dict:
                continue
            try:
                year = int(val_dict['year'])
            except ValueError:
                continue
            # Apply conditions
            if year <= max_year and year >= min_year:
                filtered_corpus[key] = val_dict

        self.corpus_meta = filtered_corpus

        print(f'Corpus filtered. {orig_len - len(self.corpus_meta)} documents removed.')

    def filter_by_type(self, allowed_types):
        """Filters the corpus by doctype.

        Arguments:
        - allowed_types (list): a list of strings with the allowed doc_types"""

        filtered_corpus = {}

        orig_len = len(self)
        print(f'Filtering {orig_len} documents ...')

        for key, val_dict in self.corpus_meta.items():
            if val_dict['type'] in allowed_types:
                filtered_corpus[key] = val_dict

        self.corpus_meta = filtered_corpus

        print(f'Corpus filtered. {orig_len - len(self.corpus_meta)} documents removed.')

    def save(self, path=None):
        """Pickles the corpus metadata for later use.

        Arguments:
        path (str): path to the save file"""

        if path is None:
            path = time.strftime("%Y%m%d-%H%M%S") + '-jstor-corpus.p'

        out = {'meta_dir':self.meta_dir, 'data_dir':self.data_dir, 'corpus_meta':self.corpus_meta}

        with open(path, 'wb') as file:
            p.dump(out, file)

        print(f'Corpus saved to {path}')

    @classmethod
    def load(cls, path):
        """Load a pickled corpus created by JSTORCorpus.save()

        Arguments:
        path (str): path to the corpus"""

        with open(path, 'rb') as corpus_file:
            corpus = cls(**p.load(corpus_file))

        print(f'Corpus loaded from {path}')

        return corpus

class TargetedCollocationFinder(nltk.collocations.AbstractCollocationFinder):
    """Finds associations for a particular word, can distinguish linguistic contexts.

    The main purpose of this class is to investigate the collocations of particular words
    in a large corpus in different linguistic environments."""
    def __init__(self, word_fd, bigram_fd, target, window_size=2, include=None, exclude=None):
        """Construct a TargetedCollocationFinder, given FreqDists for
        appearances of words and (possibly non-contiguous) bigrams.

        Arguments:
        ==========
        word_fd, bigram_fd : FreqDist
            the FreqDists for the words in the corpus and the bigrams
        window_size : int
            the size of the sliding window in which collocations are found
        target : str
            the word whose collocations we are searching for
        include, exclude : list or tuple of str
            context words which must or must not appear in the window for bigram to be kept
        """
        super().__init__(word_fd, bigram_fd)
        self.window_size = window_size
        self.target = target
        self.include = include
        self.exclude = exclude

    @classmethod
    def from_words(cls, words, target, include=None, exclude=None, window_size=2):
        """Construct a TargetedCollocationFinder for all bigrams in the given
        sequence. When window_size > 2, count non-contiguous bigrams, with the option
        of keeping only those bigrams where certain context words appears in the window."""

        wfd = FreqDist()
        bfd = FreqDist()

        if window_size < 2:
            raise ValueError("Specify window_size at least 2")
        if include is not None or exclude is not None:
            if window_size < 3:
                raise ValueError("When searching with a context, specify window_size at least 3")
        if include is not None and not isinstance(include, (list, tuple)):
            raise TypeError("include must be a list or tuple")
        if exclude is not None and not isinstance(exclude, (list, tuple)):
            raise TypeError("exclude must be a list or tuple")

        for window in ngrams(words, window_size, pad_right=True):

            # As the window slides through the text, count the first word
            # each time to get the individual word frequencies
            w1 = window[0]
            if w1 is None:
                continue
            wfd[w1] += 1

            # If context is being used, check that this window is valid
            if include is not None:
                inc_score = 0
                for inc in include:
                    if inc in window:
                        inc_score += 1
                if inc_score < 1:
                    continue
            if exclude is not None:
                exc_score = 0
                for exc in exclude:
                    if exc not in window:
                        exc_score += 1
                if exc_score < 1:
                    continue

            # Collect bigram frequencies if target is in the bigram
            if w1 == target:
                for w2 in window[1:]:
                    if w2 is not None:
                        bfd[(w1, w2)] += 1
            else:
                for w2 in window[1:]:
                    if w2 == target:
                        bfd[(w1, w2)] += 1

        return cls(wfd, bfd, target, window_size, include, exclude)

    @classmethod
    def from_corpus(cls, corpus, target, include=None, exclude=None, window_size=2):
        """Construct a collocation finder given a corpus of documents,
        each of which is a list (or iterable) of tokens.
        """
        # Pad the documents to the right so that they won't overlap when windowed
        corpus_chain = cls._build_new_documents(corpus, window_size, pad_right=True)
        # Construct finder from stream of tokens
        return cls.from_words(corpus_chain, target, include, exclude, window_size)

    def score_ngram(self, score_fn, w1, w2):
        """Returns the score for a given bigram using the given scoring
        function.  Following Church and Hanks (1990), counts are scaled by
        a factor of 1/(window_size - 1).

        This function is copied from nltk.collocations.BigramCollocationFinder
        """
        n_all = self.N
        n_ii = self.ngram_fd[(w1, w2)] / (self.window_size - 1.0)
        if not n_ii:
            return
        n_ix = self.word_fd[w1]
        n_xi = self.word_fd[w2]
        return score_fn(n_ii, (n_ix, n_xi), n_all)

class CorpusBigramCollocationFinder(nltk.collocations.BigramCollocationFinder):
    """Wrapper around BigramCollocationFinder that provides a from_corpus method."""

    @classmethod
    def from_corpus(cls, corpus, window_size=2):
        """Construct a collocation finder given a corpus of documents,
        each of which is a list (or iterable) of tokens.

        This is the same method as AbstractCollocationFinder.from_documents,
        except it allows you to choose the window_size.
        """
        # Pad the documents to the right so that they won't overlap when windowed
        # Then chain them
        corpus_chain = cls._build_new_documents(corpus, window_size, pad_right=True)
        # Construct finder from stream of tokens
        return cls.from_words(corpus_chain, window_size)
