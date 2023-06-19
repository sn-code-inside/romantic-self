"""Sentiment analysers for Chapter 6

NB: There is a bug in rpy2 that prevents the RAnalyser classes from working on an M1 Mac
see https://github.com/rpy2/rpy2/issues/900"""
from abc import ABC, abstractmethod
from functools import partial
from typing import MutableSequence
from nltk.sentiment import SentimentIntensityAnalyzer
from rpy2.robjects.packages import importr

_SYUZHET_PACKAGE = importr("syuzhet")


def get_syuzhet_sentiment(sentences: list[str], method: str) -> MutableSequence[float]:
    """The get_sentiment() function from syuzhet"""
    return _SYUZHET_PACKAGE.get_sentiment(sentences, method)


class Analyser(ABC):
    """A named sentiment analysis function"""
    _model_name: str

    @property
    def model_name(self):
        """The name of the sentiment analysis model used by this anlayser"""
        return self._model_name

    @abstractmethod
    def __call__(self, sentences: list[str]) -> MutableSequence[float]:
        ...


class Vader(Analyser):
    """VADER sentiment analyzer from nltk package"""

    def __init__(self) -> None:
        self._model_name = "vader"
        self._analyser = SentimentIntensityAnalyzer()

    def __call__(self, sentences):
        return [self._analyser.polarity_scores(sentence)["compound"] for sentence in sentences]


class RAnalyser(Analyser):
    """An analyser from Matthew Jockers's syuzhet package"""

    def __init__(self, method: str) -> None:
        self._model_name = method
        self._analyser = partial(get_syuzhet_sentiment, method=method)

    def __call__(self, sentences):
        return self._analyser(sentences)


class Syuzhet(RAnalyser):
    """Matthew Jocker's literary lexicon-based analyser"""

    def __init__(self) -> None:
        super().__init__("syuzhet")


class Bing(RAnalyser):
    """The bing analyser included in syuzhet"""

    def __init__(self) -> None:
        super().__init__("bing")


class Afinn(RAnalyser):
    """The afinn analyser included in syuzhet"""

    def __init__(self) -> None:
        super().__init__("afinn")


class OpenNLP(RAnalyser):
    """The Stanford OpenNLP analyser included in syuzhet"""

    def __init__(self) -> None:
        super().__init__("stanford")
