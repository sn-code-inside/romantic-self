from abc import abstractmethod, ABC
from functools import cached_property, singledispatch
from itertools import chain
import os
import json
import re
from typing import Callable, Iterable, NamedTuple, Protocol, TypeAlias
from lxml import etree
from romanticself.utils import clean_text
from nltk.tokenize import sent_tokenize

class Paragraph(NamedTuple):
    author: str
    text: str

class Sentence(NamedTuple):
    author: str
    text: str

class Biography(ABC):
    @property
    @abstractmethod
    def author(self) -> str:
        ...
    
    @property
    @abstractmethod
    def author_id(self) -> str:
        ...
    
    @property
    @abstractmethod
    def title(self) -> str:
        ...

    @property
    @abstractmethod
    def date(self) -> str:
        ...
    
    @property
    @abstractmethod
    def sentences(self) -> Iterable[Sentence]:
        ...

    @abstractmethod
    def iter_tokens(self, allowed_authors: Iterable[str]) -> Iterable[str]:
        ...
    
    def __repr__(self):
        return f"{type(self).__name__}({self.author}, {self.title}, {self.date})"

class TextBiography(Biography):
    """Text file biography."""

    def __init__(self, path: str | list[str], manifest: str, tokenizer: Callable[[str], list[str]]):
        self.path = path
        self.tokenizer = tokenizer
        self._manifest = manifest

    @property
    def manifest(self) -> dict:
        with open(self._manifest, "rt", encoding="utf-8") as manifest_file:
            return json.load(manifest_file)
        
    @property
    def filename(self) -> str:
        """The filename of a single file, or the characteristic name of a multi-file biography"""
        if isinstance(self.path, str):
            return os.path.basename(self.path)
        elif isinstance(self.path, list):
            first_file = os.path.basename(self.path[0])
            return first_file.split("-")[0] + ".txt"
        else:
            raise TypeError
    
    @property
    def author(self) -> str:
        return self.manifest[self.filename]["author"]
    
    @property
    def author_id(self) -> str:
        return self.author

    @property
    def date(self) -> str:
        return self.manifest[self.filename]["date"]
    
    @property
    def title(self) -> str:
        return self.manifest[self.filename]["title"]

    @cached_property
    def text(self) -> str:
        paths = [self.path] if isinstance(self.path, str) else self.path
        text = ""
        for path in paths:
            with open(path, "rt", encoding="utf-8") as text_file:
                raw = text_file.read()
                text += clean_text(raw)
                text += " "
        return text
    
    @property
    def sentences(self) -> Iterable[Sentence]:
        return [Sentence(self.author, sent) for sent in sent_tokenize(self.text)]
    
    def iter_tokens(self, allowed_authors = None) -> list[str]:
        del allowed_authors
        return self.tokenizer(self.text)

class XMLBiography(Biography):
    """Lives-and-letters biography from lordbyron.org"""

    NS = {
        "tei": "http://www.tei-c.org/ns/1.0",
        "xml": "http://www.w3.org/XML/1998/namespace"
    }

    def __init__(self, path: str, tokenizer: Callable[[str], list[str]]):
        self.path = path
        self.parser = etree.XMLParser()
        self._paragraphs: list[Paragraph] = []
        self._floating_author_dict: dict[str, str] = {}
        self.tokenizer = tokenizer

    @cached_property
    def tree(self) -> etree._ElementTree:
        """Element tree of the underlying xml file"""
        with open(self.path, 'rb') as xml_bytes:
            tree = etree.parse(xml_bytes, self.parser)
        return tree

    @cached_property
    def author_id(self) -> str:
        """The author of the biography: an id code"""
        author_node = self.find(".//tei:author")
        if author_node is None or "key" not in author_node.attrib:
            raise ValueError("No author id!")
        return author_node.attrib["key"]  # type: ignore

    @property
    def author(self) -> str:
        """The full name of the author"""
        author_node = self.find(".//tei:author")
        if author_node is None or author_node.text is None:
            raise ValueError("No author name!")
        return author_node.text

    @property
    def date(self) -> str:
        """Publication date"""
        date_node = self.find(".//tei:sourceDesc/tei:bibl/tei:date")
        if date_node is None or "when" not in date_node.attrib:
            raise ValueError("No date!")
        return date_node.attrib["when"]  # type: ignore

    @property
    def title(self) -> str:
        """Title of the book"""
        title_node = self.find(".//tei:title")
        if title_node is None or title_node.text is None:
            raise ValueError("No title!")
        return title_node.text

    @property
    def sentences(self) -> Iterable[Sentence]:
        """All the sentences in the biography"""
        return chain(*[self._split_paragraph(para) for para in self.paragraphs])
    
    def iter_tokens(self, allowed_authors: Iterable[str] | None = None) -> Iterable[str]:
        def _true(_):
            return True
        def _in_allowed(author):
            return author in allowed_authors
        predicate = _true if allowed_authors is None else _in_allowed
        return chain(*(self.tokenizer(paragraph.text)
                       for paragraph in self.paragraphs
                       if predicate(paragraph.author)))

    @property
    def paragraphs(self) -> list[Paragraph]:
        """All the paragraphs in the biograpy, labelled by author"""
        if not self._paragraphs:
            text_node = self.find(".//tei:text")
            if text_node is None:
                raise ValueError("Tree has no text node")
            self._get_paragraphs(text_node, author=self.author_id)
        return self._paragraphs
    
    @cached_property
    def all_authors(self) -> set[str]:
        return set(para.author for para in self.paragraphs)

    @staticmethod
    def _split_paragraph(paragraph: Paragraph) -> list[Sentence]:
        auth = paragraph.author
        return [Sentence(auth, sent) for sent in sent_tokenize(paragraph.text)]

    def _get_paragraphs(self, elem: etree._Element, author: str) -> None:
        match etree.QName(elem).localname:
            case "p":
                text = etree.tostring(elem, method="text", encoding="unicode")
                self._paragraphs.append(Paragraph(author, clean_text(text, gutenberg = False)))
            case "floatingText":
                new_author = self._get_floating_author(elem)
                for child in elem:
                    self._get_paragraphs(child, new_author)
            case _:
                for child in elem:
                    self._get_paragraphs(child, author)

    def _get_floating_author(self, elem: etree._Element) -> str:
        ft_id = self._get_floating_text_id(elem)
        if ft_id in self._floating_author_dict:
            return self._floating_author_dict[ft_id]
        elif (author_node := elem.find(".//tei:docAuthor", namespaces=self.NS)) is not None:
            author_id: str = author_node.attrib["n"]  # type: ignore
            self._floating_author_dict[ft_id] = author_id
            return author_id
        else:
            return "Unknown"

    def _get_floating_text_id(self, elem: etree._Element) -> str:
        id_rgx = re.compile(r"\w+\.\d+")
        first_div = elem.find(".//tei:div", namespaces=self.NS)
        # type: ignore
        div_id: str = first_div.attrib[f"{{{self.NS['xml']}}}id"] #type: ignore
        id_match = id_rgx.match(div_id)
        return id_match.group(0) if id_match is not None else ""

    def find(self, path: str) -> etree._Element | None:
        """Get first match for `path`. Automatically namespaced to tei"""
        return self.tree.find(path, namespaces=self.NS)
