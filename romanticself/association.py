"""Utilities for accessing JSTOR Data for Research Corpus"""

import itertools as _itertools
import re as _re

from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.collocations import AbstractCollocationFinder, BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.corpus import stopwords

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

class CorpusBigramCollocationFinder(BigramCollocationFinder):
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

class RobustBigramAssocMeasures(BigramAssocMeasures):
    """Bigram association measures with a modified _contingency
    method which disallows negative values. This is a problem
    when the window_size is large. When the window is wide, certain words may
    never appear without a particular collocate. In this case, the observed
    frequency of the word *without* that collocate may be negative.
    """
    @staticmethod
    def _contingency(n_ii, n_ix_xi_tuple, n_xx):
        """Calculates values of a bigram contingency table from marginal values."""
        (n_ix, n_xi) = n_ix_xi_tuple
        n_oi = max(n_xi - n_ii, 0)
        n_io = max(n_ix - n_ii, 0)
        return (n_ii, n_oi, n_io, n_xx - n_ii - n_oi - n_io)

class RobustTrigramAssocMeasures(TrigramAssocMeasures):
    """Trigram association measures with a modified _contingency
    method which disallows negative values. This is a problem
    when the window_size is large. When the window is wide, certain words may
    never appear without particular collocates. In this case, the observed
    frequency of the word *without* that collocate may be negative.
    """
    @staticmethod
    def _contingency(n_iii, n_iix_tuple, n_ixx_tuple, n_xxx):
        """Calculates values of a trigram contingency table (or cube) from
        marginal values.
        >>> TrigramAssocMeasures._contingency(1, (1, 1, 1), (1, 73, 1), 2000)
        (1, 0, 0, 0, 0, 72, 0, 1927)
        """
        (n_iix, n_ixi, n_xii) = n_iix_tuple
        (n_ixx, n_xix, n_xxi) = n_ixx_tuple
        n_oii = max(n_xii - n_iii, 0)
        n_ioi = max(n_ixi - n_iii, 0)
        n_iio = max(n_iix - n_iii, 0)
        n_ooi = max(n_xxi - n_iii - n_oii - n_ioi, 0)
        n_oio = max(n_xix - n_iii - n_oii - n_iio, 0)
        n_ioo = max(n_ixx - n_iii - n_ioi - n_iio, 0)
        n_ooo = max(n_xxx - n_iii - n_oii - n_ioi - n_iio - n_ooi - n_oio - n_ioo, 0)

        return (n_iii, n_oii, n_ioi, n_ooi, n_iio, n_oio, n_ioo, n_ooo)

def stopword_filter(*ngram):
    """Returns true if the ngram contains junk or a stopword"""

    sw = set(stopwords.words('english'))

    if not sw.isdisjoint(ngram):
        return True
    elif any([_re.match(r'[\W\d]+', wd) for wd in ngram]):
        return True
    else:
        return False