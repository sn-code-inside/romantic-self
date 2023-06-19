import os
import re

# For normalisation of tokens
WORD_RGX = re.compile('[A-Za-z]')

# For cleaning Gutenberg files
GUT_HEADER_RGX = re.compile(
    r'\A.+\*{3} {0,2}START OF.{,200}\*{3}', flags=re.DOTALL)
GUT_LICENCE_RGX = re.compile(r'\*{3} {0,2}END OF.+', flags=re.DOTALL)

# Other cleaning
WHITESPACE_RGX = re.compile(r"\s+")


def import_scheme_map(path: str) -> dict:
    """Import rhyme scheme map from csv file"""

    scheme_map = {}

    with open(path, "rt", encoding="utf8") as scheme_file:
        for line in scheme_file.readlines()[1:]:
            scheme, code = line.split(",")
            scheme_map[scheme] = code.strip()

    return scheme_map


def strip_sonnet_numbering(string: str) -> str:
    """Strips sonnet numbering from sonnet titles. For the specific analysis
    in Chapter 4 of the book, Clare, Smith and Wordsworth all use different numbering
    practices:

    - Clare: no numbering
    - Smith: all sonnets are numbered 'SONNET I.', 'SONNET II.' ...
    - Wordsworth: all sonnets are numbered 'I.', 'II.' etc.

    The numbering must be stripped to make the titles comparable."""
    sonnet_numbering = re.compile(
        r'^\s?(SONNET\s)?M{0,3}(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[VX]|V?I{0,3})\.?\s+')
    return sonnet_numbering.sub('', string)


def file_ext(path: str) -> str:
    """The file extension of the passed path"""
    return os.path.splitext(path)[1]


def clean_text(text: str, gutenberg=True) -> str:
    """Apply basic normalisation to raw text, e.g. removes Gutenberg boilerplate
    and duplicated whitespace."""
    if gutenberg is True:
        text = GUT_HEADER_RGX.sub("", text)
        text = GUT_LICENCE_RGX.sub("", text)
    text = WHITESPACE_RGX.sub(" ", text)
    text = text.strip()
    return text
