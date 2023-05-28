from functools import cached_property, singledispatchmethod
import re
from lxml import etree

class TextBiography():
    """Text file biography."""

class XMLBiography():
    """Lives-and-letters biography from lordbyron.org"""

    NS = {
        "tei": "http://www.tei-c.org/ns/1.0",
        "xml": "http://www.w3.org/XML/1998/namespace"
        }

    def __init__(self, path):
        self.path = path
        self.parser = etree.XMLParser()
        self._paragraphs = []
        self._floating_author_dict = {}
    
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
        return author_node.attrib["key"] #type: ignore
    
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
        return date_node.attrib["when"] #type: ignore
    
    @property
    def title(self) -> str:
        """Title of the book"""
        title_node = self.find(".//tei:title")
        if title_node is None or title_node.text is None:
            raise ValueError("No title!")
        return title_node.text

    @cached_property
    def sentences(self):
        """All the sentences in the biography"""
        raise NotImplementedError
        
    @property
    def paragraphs(self):
        """All the paragraphs in the biograpy, labelled by author"""
        if not self._paragraphs:
            text_node = self.find(".//tei:text")
            if text_node is None:
                raise ValueError("Tree has no text node")
            self._get_paragraphs(text_node, author = self.author_id)
        return self._paragraphs
        
    def _get_paragraphs(self, elem: etree._Element, author: str):
        match etree.QName(elem).localname:
            case "p":
                self._paragraphs.append(
                    (author,
                     etree.tostring(elem, method="text", encoding="unicode"),)
                    )
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
            author_id: str = author_node.attrib["n"] #type: ignore
            self._floating_author_dict[ft_id] = author_id
            return author_id
        else:
            return "Unknown"
        
    def _get_floating_text_id(self, elem: etree._Element) -> str:
        id_rgx = re.compile(r"\w+\.\d+")
        first_div = elem.find(".//tei:div", namespaces=self.NS)
        if first_div is None:
            breakpoint()
        div_id: str = first_div.attrib[f"{{{self.NS['xml']}}}id"] #type: ignore
        id_match = id_rgx.match(div_id)
        return id_match.group(0) if id_match is not None else ""
        
    def find(self, path: str) -> etree._Element | None:
        """Get first match for `path`. Automatically namespaced to tei"""
        return self.tree.find(path, namespaces=self.NS)
