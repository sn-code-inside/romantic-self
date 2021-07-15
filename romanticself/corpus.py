"""Utilities for accessing Research Corpora"""

import os
import re
import time
import pickle as p
import json
from math import inf
from itertools import chain

from bs4 import BeautifulSoup
from nltk import word_tokenize, wordpunct_tokenize, pos_tag_sents
from nltk.corpus import stopwords
import nltk

class JSTORCorpus(object):
    """Iterator for streaming files. Also allows basic filtering.

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
                journal = meta_xml.find('journal-title')
                if journal is not None:
                    doc_dict['journal'] = journal.get_text()

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

class NovelCorpus(object):
    """Iterator for loading and tokenising files.

    NB: Unliked the JSTORCorpus class, this class loads the texts into memory,
    so is only appropriate for relatively small corpora (it was developed for a
    project using only 40 novels).
    
    Arguments:
    - data_dir (str): path to txt files
    - tokenizer (fn): tokenizer of choice. Defaults to nltk.tokenize.word_tokenize"""

    # For cleaning Gutenberg files
    GUT_HEADER_RGX = re.compile(r'\A.+\*{3} {0,2}START OF.{,200}\*{3}', flags = re.DOTALL)
    GUT_LICENCE_RGX = re.compile(r'\*{3} {0,2}END OF.+', flags = re.DOTALL)

    # For normalisation
    WORD_RGX = re.compile('[A-Za-z]')

    def __init__(self, data_dir, tokenizer=word_tokenize):
        self.data_dir = data_dir
        self.manifest_pth = data_dir + "/manifest.json"
        self.data = dict()
        self.tokenizer = tokenizer

        # Import manifest file
        with open(self.manifest_pth, "rt") as file:
            self.data = json.load(file)
        
        # Import and tokenize novels
        for key in self.data:
            text = self._read_normalise(os.path.join(self.data_dir, key))
            tokens = [tk for tk in self.tokenizer(text) if self.WORD_RGX.match(tk)]
            self.data[key]["tokens"] = tokens

        # Print import message
        print(f"{len(self)} novels imported from {self.data_dir}.")
    
    def __iter__(self):
        """Yields tokenized texts from the corpus"""
        for key in self.data:
            # Yield tokens
            yield self.data[key]["tokens"]
    
    def __len__(self):
        return len(self.data)

    def to_csv(self, out_pth="novel_corpus_summary.csv"):
        csv = "Title,Author,Year,Nation,Gothic,Network,Source,Available Online\n"

        def _src(url):
            if url.startswith("http://hdl.handle.net"):
                return "Oxford Text Archive"
            if url.startswith("https://gutenberg.org"):
                return "Project Gutenberg"
            if url.startswith("https://www.proquest.com"):
                return "Literature Online (Proquest)"
            if url.startswith("https://en.wikisource.org"):
                return "Wikisource"
            else:
                raise ValueError("Source not known.")

        def _avail(licence):
            if licence == "Restrictive":
                return "No"
            else:
                return "Yes"

        def _bool2str(bool_val):
            if bool_val:
                return "Yes"
            else:
                return "No"

        sorted_manifest = sorted(self.data.values(), key=lambda x: x["year"])

        for novel in sorted_manifest:
            # Write to csv
            csv += novel["short_title"] + ","
            csv += novel["author"] + ","
            csv += str(novel["year"]) + ","
            csv += novel["nation"] + ","
            csv += _bool2str(novel["gothic"]) + ","
            csv += _bool2str(novel["network"]) + ","
            if isinstance(novel["source"], list):
                csv += _src(novel["source"][0]) + ","
            else:
                csv += _src(novel["source"]) + ","
            csv += _avail(novel["licence"])
            csv += "\n"

        with open(out_pth, "wt") as file:
            file.write(csv)

    def iter_filter(self, **kwargs):
        """Iterates over parts of the corpus, defined by filter. Only novels with the
        attributes specified in **kwargs will be included.

        e.g. corpus.iter_filter(year=1799) will return only novels published in 1799
        """
        for key in self.data:

            skip = False

            # Check each filter keyword
            for filter,val in kwargs.items(): 
                if self.data[key][filter] != val:
                    skip = True

            if skip:
                continue

            # Yield tokens
            yield self.data[key]["tokens"]

    def _read_normalise(self, text_path):
        """Reads in text file and normalises it.
        
        Arguments:
        - text_path (str): path to text file"""
        
        # Import text
        with open(text_path, mode="rt", errors="ignore") as file:
            text = file.read()

        # Strip gutenberg bufferplate
        text = self.GUT_HEADER_RGX.sub("", text)
        text = self.GUT_LICENCE_RGX.sub("", text)

        # Normalise
        text = text.lower()
        text = re.sub(r'_(?=\w)', '', text) # Strip underscores from before words
        text = re.sub(r'(?<=(\w))_', ' ', text) # And after
        text = re.sub(r' \d+(th|rd|nd|st|mo|\W+)\b', ' ', text) # Also drop numbers 

        return text

    def yield_metadata(self, *args):
        """Yields requested metadata about novels in corpus.
        
        Arguments:
        - *args (str): metadata values to return"""
        
        requested_metadata = []

        for val in self.data.values():
            requested_metadata.append(tuple(val[arg] for arg in args))
        
        return requested_metadata

class NovelPOSCorpus(NovelCorpus):
    """Iterator for Novel corpus, which yield part-of-speech-tagged tokens instead of raw tokens.

    Only common nouns, adjectives, adverbs and verbs are retained. Common stopwords removed.
    
    Arguments:
    - data_dir (str): path to txt files
    - tokenizer (fn): tokenizer of choice. Defaults to nltk.word_tokenize"""

    # Regex to filter punctuation out of tokens
    WORD_POS_RGX = re.compile('[A-Za-z]+_.+')

    # Mapping for UPenn tags
    TAG_MAP = {
        "JJ":"adjective",
        "JJR":"adjective",
        "JJS":"adjective",
        "NN":"noun",
        "NNS":"noun",
        "RB":"adverb",
        "RBR":"adverb",
        "RBS":"adverb",
        "VB":"verb",
        "VBD":"verb",
        "VBG":"verb",
        "VBN":"verb",
        "VBP":"verb",
        "VBZ":"verb"
    }

    def __init__(self, data_dir, tokenizer=word_tokenize):
        self.data_dir = data_dir
        self.manifest_pth = data_dir + "/manifest.json"
        self.data = dict()
        self.tokenizer = tokenizer

        # Import manifest file
        with open(self.manifest_pth, "rt") as file:
            self.data = json.load(file)
        
        print(f"Found {len(self)} novels in {self.data_dir}.")

        # Import and tokenize novels
        iter_count = 0
        for key in self.data:
            tokens = self._read_tag(os.path.join(self.data_dir, key))
            self.data[key]["tokens"] = tokens
            
            iter_count += 1
            if iter_count % 10 == 0:
                print(f"{iter_count} novels imported ...")

        # Print import message
        print(f"{len(self)} novels imported, tagged and tokenised from {self.data_dir}.")
    
    def _read_tag(self, text_path):
        """Reads in text file, tags and normalises it. Only common nouns, adjectives,
        adverbs and verbs are retained.
        
        Arguments:
        - text_path (str): path to text file"""

        eng_stops = set(stopwords.words("english"))
        
        # Import text
        with open(text_path, mode="rt", errors="ignore") as file:
            text = file.read()

        # Strip gutenberg bufferplate
        text = self.GUT_HEADER_RGX.sub("", text)
        text = self.GUT_LICENCE_RGX.sub("", text)

        # Normalise
        text = re.sub(r'_(?=\w)', '', text) # Strip underscores from before words
        text = re.sub(r'(?<=(\w))_', ' ', text) # And after

        # Split into sentences
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_detector.tokenize(text)

        # Tokenise and tag
        sentences = [self.tokenizer(sent) for sent in sentences]
        sentences = pos_tag_sents(sentences)

        # Reformat and filter tokens
        tokens = []
        for token,tag in chain(*sentences):
            token = token.lower()
            if token in eng_stops:
                continue
            if tag in self.TAG_MAP:
                tokens.append(token + "_" + self.TAG_MAP[tag])

        return tokens

class SonnetCorpus(object):
    """Iterator for streaming sonnet files."""

    def __init__(self):
        return NotImplemented

def ota_xml_to_txt(dir="."):
    """Helper function that converts OTA xml files into raw text. If you have
    downloaded several files for a multi-volume work, and wish to concatenate them,
    you will need to do this by hand.
    
    Arguments:
    - dir (str): the directory where the files are held, and where the new
                .txt files will be written."""

    xml_files = [file for file in os.listdir(dir) if file.endswith(".xml")]

    print(f"Files found: {', '.join(xml_files)}")

    for xml_path in xml_files:
        with open(xml_path, mode="rt") as file:
            soup = BeautifulSoup(file, features="html.parser")
        text = soup.text
        text = text.replace("ſ", "s")
        txt_path = xml_path[:-4] + ".txt"
        with open(txt_path, mode="wt") as file:
            file.write(text)
        
        print(f"Text from {xml_path} written to {txt_path}")