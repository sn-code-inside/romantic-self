"""Utilities for accessing JSTOR Data for Research Corpus"""

import os
import re
import time
import pickle as p
from math import inf
import itertools as _itertools

from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.collocations import AbstractCollocationFinder, BigramCollocationFinder

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

class AbstractTargetedAssocFinder(AbstractCollocationFinder):
    """A modified abstract base class for the TargetedAssocFinders. It
    maintains the structure of nltk.collocations.AbstractCollocationFinder,
    but provides modified methods for building a finder from a corpus
    of documents."""

    @classmethod
    def _build_new_documents(
            cls, documents, window_size, pad_left=True, pad_right=True, pad_symbol=None
        ):
        """
        Pad the document with the place holder according to the window_size
        """
        padding = (pad_symbol,) * (window_size - 1)
        if pad_right and pad_left:
            return _itertools.chain.from_iterable(
                _itertools.chain(padding, doc, padding) for doc in documents
            )
        elif pad_left:
            return _itertools.chain.from_iterable(
                _itertools.chain(padding, doc) for doc in documents
            )
        elif pad_right:
            return _itertools.chain.from_iterable(
                _itertools.chain(doc, padding) for doc in documents
            )

class TargetedBigramAssocFinder(AbstractTargetedAssocFinder):
    """Finds associations for a particular word, can distinguish linguistic contexts.

    The main purpose of this class is to investigate the collocations of particular words
    in a large corpus in different linguistic environments."""

    default_ws = 3

    def __init__(self, word_fd, bigram_fd, target, window_size=3):
        """Construct a TargetedCollocationFinder, given FreqDists for
        appearances of words and (possibly non-contiguous) bigrams.

        Arguments:
        ==========
        word_fd, bigram_fd : FreqDist
            the FreqDists for the words in the corpus and the bigrams
        window_size : int > 3
            the size of the sliding window in which collocations are found. Must
            be odd.
        target : str
            the target word whose collocations are sought
        """
        super().__init__(word_fd, bigram_fd)
        self.window_size = window_size
        self.target = target

    @classmethod
    def from_words(cls, words, target, window_size=3):
        """Construct a TargetedCollocationFinder for all bigrams in the given
        sequence. The purpose is to find possible collocates for the target
        word. The TargetedCollocationFinder looks for unordered associations,
        rather than ordered ones as the BigramCollocationFinder does."""

        wfd = FreqDist()
        bfd = FreqDist()

        if window_size < 3:
            raise ValueError("Specify window_size at least 3")
        if (window_size % 2) != 1:
            raise ValueError("window_size must be odd")
        if not isinstance(target, str):
            raise TypeError("target must be a string")

        ctr = int(window_size / 2)

        for window in ngrams(words, window_size, pad_left=True, pad_right=True):

            # Get central word in window. Skip if it is left-padding
            w1 = window[ctr]
            if w1 is None:
                continue
            wfd[w1] += 1

            # Skip if w1 is target
            if w1 is target:
                continue
            # Otherwise, if the target is in the window, count the bigram
            if target in window:
                bfd[(w1, target)] += 1

        return cls(wfd, bfd, window_size, target)

    @classmethod
    def from_corpus(cls, corpus, target, window_size=3):
        """Construct a collocation finder given a corpus of documents,
        each of which is a list (or iterable) of tokens.

        Arguments:
        ==========
        corpus : iterable of iterables of str
            The corpus of documents in which ngrams are to be found. Each
            document should be an iterable of tokens.
        target : str
            The target word, which must appear in the bigrams
        window_size : int
            The width of the search window. Must be odd and > 3.
        """
        # Pad the documents to the right so that they won't overlap when windowed
        corpus_chain = cls._build_new_documents(corpus, window_size)
        # Construct finder from stream of tokens
        return cls.from_words(corpus_chain, target, window_size)

    def score_ngram(self, score_fn, w1, w2):
        """Returns the score for a given bigram using the given scoring
        function. No scaling is applied, contra Church and Hanks (1990).
        """
        n_all = self.N
        n_ii = self.ngram_fd[(w1, w2)]
        if not n_ii:
            return None
        n_ix = self.word_fd[w1]
        n_xi = self.word_fd[w2]
        return score_fn(n_ii, (n_ix, n_xi), n_all)

class TargetedTrigramAssocFinder(AbstractTargetedAssocFinder):
    """Finds trigrams that include or exclude particular words.

    The main purpose of this class is to enable trigram search in very large corpora,
    where the standard TrigramCollocationFinder may exceed memory."""

    default_ws = 3

    def __init__(self, word_fd, bigram_fd, trigram_fd, targets, window_size=3):
        """Construct a TargetedTrigramCollocationFinder, given FreqDists for
        appearances of words, bigrams, and trigrams.
        """
        super().__init__(word_fd, trigram_fd)
        self.bigram_fd = bigram_fd
        self.targets = targets
        self.window_size = window_size

    @classmethod
    def from_words(cls, words, targets, window_size=3):
        """Construct a TrigramCollocationFinder for all trigrams in the given
        sequence, filtering for 2 'include' words.

        Arguments:
        ==========
        words : iterable of str
            the sequence of words in which to search for trigrams
        targets : list, tuple or set of str
            a sequence of 2 words which must appear in the trigram
            for it to be counted.
        window_size : int >= 3, must be odd
            the size of the search window, must be odd
        """

        if window_size < 3:
            raise ValueError("Specify window_size at least 3")
        if window_size % 2 != 1:
            raise ValueError("window_size must be odd")
        if not isinstance(targets, (set, tuple, list)):
            raise TypeError("targets must be a set, tuple or list")
        if len(targets) != 2:
            raise ValueError("targets must contain two words")

        wfd = FreqDist()
        bfd = FreqDist()
        tfd = FreqDist()
        _targets = set(targets)
        tar_1, tar_2 = _targets
        ctr = int(window_size / 2)

        # in this implementation, we pad both left and right
        # we also don't respect the order of the ngrams
        for window in ngrams(words, window_size, pad_left=True, pad_right=True):
            # Get central word in the window
            ctr_word = window[ctr]
            if ctr_word is None:
                continue
            wfd[ctr_word] += 1
            # If target words are present in the window, count
            # the necessary ngrams
            present_targets = _targets.intersection(window)
            # If any of the target words are present, count the bigrams
            # This will also count instances of (tar_1, tar_2) or
            # (tar_2, tar_1)
            if len(present_targets) > 0:
                for tar in present_targets:
                    bfd[(ctr_word, tar)] += 1
            # If both targets are present, count the trigram
            if len(present_targets) == 2:
                tfd[(ctr_word, tar_1, tar_2)] += 1

        # Combine counts for (tar_1, tar_2) with (tar_2, tar_1)
        bfd[(tar_1, tar_2)] += bfd[(tar_2, tar_1)]
        del bfd[(tar_2, tar_1)]

        return cls(wfd, bfd, tfd, targets, window_size)

    @classmethod
    def from_corpus(cls, corpus, targets, window_size=3):
        """Construct a collocation finder given a corpus of documents,
        each of which is a list (or iterable) of tokens.

        Arguments:
        ==========
        corpus : iterable of iterables of str
            The corpus of documents in which ngrams are to be found. Each
            document should be an iterable of tokens.
        targets : list, tuple or set of str
            2 target words, which must appear in the trigram for it to be
            counted.
        window_size : int
            The width of the search window. Must be odd and > 3.
        """
        # Pad the documents to the right so that they won't overlap when windowed
        corpus_chain = cls._build_new_documents(corpus, window_size)
        # Construct finder from stream of tokens
        return cls.from_words(corpus_chain, targets, window_size)

    def bigram_finder(self):
        """Constructs a bigram collocation finder with the bigram and unigram
        data from this finder. The finder is effectively pre-filtered to only
        include bigrams with one or other target word.
        """
        return BigramCollocationFinder(self.word_fd, self.bigram_fd)

    def score_ngram(self, score_fn, w1, w2, w3):
        """Returns the score for a given trigram using the given scoring
        function.
        """
        n_all = self.N
        n_iii = self.ngram_fd[(w1, w2, w3)]
        if not n_iii:
            return None
        n_iix = self.bigram_fd[(w1, w2)]
        n_ixi = self.bigram_fd[(w1, w3)]
        n_xii = self.bigram_fd[(w2, w3)]
        n_ixx = self.word_fd[w1]
        n_xix = self.word_fd[w2]
        n_xxi = self.word_fd[w3]
        return score_fn(n_iii, (n_iix, n_ixi, n_xii), (n_ixx, n_xix, n_xxi), n_all)

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
