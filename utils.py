"""Utilities for accessing JSTOR Data for Research Corpus"""

import os
import re
import time
import pickle as p
from math import inf

from bs4 import BeautifulSoup
from tqdm import tqdm
from nltk.tokenize import wordpunct_tokenize

class JSTORCorpus(object):
    """Iterator for streaming files into Gensim. Also allows basic filtering.

    Arguments:
    - meta_dir (str): path to xml files
    - data_dir (str): path to txt files
    - corpus_meta (dict, optional): corpus dict, used internally"""

    TAG_RGX = re.compile('<[^>]+>') # For cleaning .txt files

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
                text = self.TAG_RGX.sub('', raw_xml)
                # Yield array of tokens
                yield wordpunct_tokenize(text)
                
    def __len__(self):
        return len(self.corpus_meta)
    
    def iter_lower(self):
        for key in self.corpus_meta:
            with open(key) as file:
                # Get text
                raw_xml = file.read()
                # Strip tags
                text = self.TAG_RGX.sub('', raw_xml)
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

        for name in tqdm(os.listdir(meta_dir)):
 
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
                title = meta_xml.find(['article-title','source'])
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
                # Getting chapter title is slightly harder, because sometimes each book-part is labelled
                # simply with the internal id, and sometimes with the doi
                book_id = re.sub('.+_', '', doi)
                book_rgx = re.compile(re.escape(book_id))
                doc_dict['title'] = meta_xml.find('book-part-id', string=book_rgx).parent.find('title').get_text()

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

        for key,val_dict in self.corpus_meta.items():
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
            if val_dict.type in allowed_types:
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

def load_jstor_corpus(path):
    """Helper function that loads a pickled corpus created by JSTORCorpus.save()

    Arguments:
    - path (str): path to the corpus"""

    with open(path, 'rb') as corpus_file:
        corpus = JSTORCorpus(**p.load(corpus_file))

    return corpus
