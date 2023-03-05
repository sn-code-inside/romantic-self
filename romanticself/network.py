"""Functions for importing and analysing network data for Chapter 5"""

from collections import defaultdict, namedtuple
import re
from typing import Iterable, TypeAlias
import os
import json
import csv
import textwrap
import igraph as ig
from igraph import VertexSeq, EdgeSeq
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

GraphCorpus: TypeAlias = dict[str, ig.Graph]
KeyedTokens: TypeAlias = dict[str, list[str]]

WORD_RGX = re.compile(r"[a-zA-Z]")


def import_network_data(data_dir: str, use_cache: bool = True) -> GraphCorpus:
    """Import graphs for each play in corpus, using raw files if no cached
    files available."""
    manifest = _import_manifest(data_dir)
    assert isinstance(use_cache, bool)
    if use_cache and _is_cache(data_dir):
        return _import_from_cache(data_dir, manifest)
    else:
        return _import_from_raw(data_dir, manifest)


def _is_cache(data_dir: str) -> bool:
    """Are there any gml files in the networks data directory?"""
    gml_files = [file for file in os.listdir(
        data_dir) if file.endswith(".gml")]
    return len(gml_files) > 0


def _import_manifest(data_dir: str) -> dict[str, dict]:
    with open(os.path.join(data_dir, "manifest.json"), "rt", encoding="utf8") as manifest:
        return json.load(manifest)


def _import_from_raw(data_dir: str, manifest: dict, cache_corpus: bool = True) -> GraphCorpus:
    """Import graphs from raw files (including copyrighted text). Saves corpus to
    disk for reuse."""
    assert isinstance(cache_corpus, bool)
    raw_dir = os.path.join(data_dir, "raw")
    corpus = dict()
    for play, metadata in manifest.items():
        edges = _dedupe_edgelist(raw_dir, play)
        graph = _get_graph_from_edgelist(edges, metadata, play)
        graph = _update_vert_attrs_from_cast(graph, raw_dir)
        graph = _collect_tokens_by_character(graph, raw_dir)
        graph = _get_character_keywords(graph)
        graph = _compute_community_structure(graph, use_weights=True)
        graph = _get_community_keywords(graph)
        corpus[play] = graph
    if cache_corpus:
        _save_corpus_as_graphml(data_dir, corpus)
    return corpus


def _save_corpus_as_graphml(data_dir: str, corpus: GraphCorpus) -> None:
    for graph in corpus.values():
        _graph = graph.copy()
        # These two lines delete unpublishable data: the dialogue for each
        # character and edge. In fact, because these data are stored as lists,
        # they are not compatible with graphml anyway and would not be saved.
        del _graph.es["tokens"]
        del _graph.vs["tokens"]
        # It is important, however, to save the `top_n_words`, which are
        # also in list format. So these attributes will need to be converted
        # to strings
        _graph = _stringify_list_attrs(_graph, "es")
        _graph = _stringify_list_attrs(_graph, "vs")
        _graph.write_graphml(_get_gml_filename(data_dir, graph["play"]))

def _stringify_list_attrs(graph: ig.Graph, which: str) -> ig.Graph:
    assert(which == "vs" or which == "es")
    vec: VertexSeq|EdgeSeq = eval(f"graph.{which}") #ignore: eval-used
    for attr in vec.attributes():
        if isinstance(vec[attr][0], list):
            vec[attr] = [repr(val) for val in vec[attr]]
    return graph

def _unstringify_list_attrs(graph: ig.Graph, which: str) -> ig.Graph:
    # This will convert any attribute that looks like a Python list into
    # one. That's 0-risk for this particular project! But beware if you
    # ever reuse this code...
    assert(which == "vs" or which == "str")
    vec: VertexSeq|EdgeSeq = eval(f"graph.{which}") #ignore: eval-used
    for attr in vec.attributes():
        try:
            eval_vec = [eval(val) for val in vec[attr]] #ignore: eval-used
            if all(isinstance(val, list) for val in eval_vec):
                vec[attr] = eval_vec
        except SyntaxError:
            continue
    return graph


def _import_from_cache(data_dir: str, manifest: dict) -> GraphCorpus:
    corpus = {play: ig.Graph.Read_GraphML(_get_gml_filename(data_dir, play)) for play in manifest}
    corpus = {play: _unstringify_list_attrs(graph, "vs") for play,graph in corpus.items()}
    corpus = {play: _unstringify_list_attrs(graph, "es") for play,graph in corpus.items()}
    return corpus


def _get_gml_filename(data_dir: str, play: str) -> str:
    return os.path.join(data_dir, f"{play}.graphml")


ImportEdge = namedtuple("ImportEdge", "source target tokens weight")


def _get_graph_from_edgelist(edges: list[ImportEdge], metadata: dict, play: str) -> ig.Graph:
    # first two fields are 'source' and 'target'; others are attrs
    edge_attrs = edges[0]._fields[2:]
    graph: ig.Graph = ig.Graph.TupleList(
        edges, directed=True, vertex_name_attr="code", edge_attrs=edge_attrs)
    graph["play"] = play
    for key, val in metadata.items():
        graph[key] = val
    return graph


def _dedupe_edgelist(raw_dir: str, play: str) -> list[ImportEdge]:
    """Combine duplicate edges by concatenating characters' speeches"""
    edges: dict[tuple[str, str], list[str]] = defaultdict(list)
    with open(os.path.join(raw_dir, f"{play}EdgeList.csv"), "rt", encoding="utf8") as edgelist:
        reader = csv.reader(edgelist)
        _ = next(reader)
        lang = "french" if play == "iphigenie" else "english"
        for src, tar, text in reader:
            edges[src, tar] += (token for token in nltk.word_tokenize(text.lower(), language=lang)
                                if WORD_RGX.match(token))
    return [ImportEdge(src, tar, tokens, len(tokens)) for (src, tar), tokens in edges.items()]


def _update_vert_attrs_from_cast(graph: ig.Graph, raw_dir: str) -> ig.Graph:
    with open(os.path.join(raw_dir, f"{graph['play']}Cast.csv"), "rt", encoding="utf8") as vertexlist:
        vertices = csv.reader(vertexlist)
        # columns after 'code' contain node attributes
        vert_attrs = next(vertices)[1:]
        for code, *attrs in vertices:
            vertex_idx = graph.vs["code"].index(code)
            for key, val in zip(vert_attrs, attrs):
                graph.vs[vertex_idx][key] = val
    return graph


def _collect_tokens_by_character(graph: ig.Graph, raw_dir: str) -> ig.Graph:
    char_tokens: KeyedTokens = defaultdict(list)
    with open(os.path.join(raw_dir, f"{graph['play']}EdgeList.csv"), "rt", encoding="utf8") as edgelist:
        reader = csv.reader(edgelist)
        _ = next(reader)
        lang = "french" if graph["play"] == "iphigenie" else "english"
        last_speech = ""
        last_src = ""
        for src, _, text in reader:
            if text != last_speech or src != last_src:
                char_tokens[src] += (token for token in nltk.word_tokenize(text.lower(), language=lang)
                                     if WORD_RGX.match(token))
                last_speech = text
                last_src = src
    graph.vs["tokens"] = [char_tokens[code] for code in graph.vs["code"]]
    graph.vs["word_count"] = [len(tokens) for tokens in graph.vs["tokens"]]
    return graph


def _get_distinctive_words(documents: Iterable[Iterable[str]], n: int = 20) -> list[list[str]]:
    """Compute n most distinctive words for passed documents

    ## A note on normalisation

    Considered as a corpus, each play is relatively small (15-25 characters)
    As a result, tf-idf does not sufficiently distinguish high-frequency words
    such as 'the' or 'and' from distinctive words for each character.
    In my PhD analysis, I used R's tm package, which accounts for this problem
    normalising the term-document matrix columnwise. In other words, it looks
    at each term, and calculates what proportion of all instances of the term occurs
    in that one text. E.g. if there are 1000 instances of "the" in the play, and
    one character says "the" 100 times, their term frequency for "the" would
    be calculated as 100/1000 = 0.1. They say 10% of the "the"s.
    
    This is not the method that nltk offers to normalise term frequencies for
    tf-idf. It offers an alternative normalisation strategy, "sublinear
    term frequency". In this strategy, the tf-idf is calculated on the log
    of the term frequency. Thus the character who uses "the" 100 times would
    have a score for this word of log_2(100) + 1 = 7.64.

    Rather than trying to re-implement the approach from my PhD thesis, I have
    selected the built-in NLTK approach. Not only is it simpler to code, but it
    also decouples the characters from one another. If a certain term is used by
    2-3 characters distinctively, this log-normalisation approach won't penalise that
    term, unlike term-column normalisation approach.
    """
    tfidf = TfidfVectorizer(tokenizer=lambda x: x,
                            lowercase=False, sublinear_tf=True)
    tfidf_matrix = tfidf.fit_transform(documents).toarray()
    sort_indices = tfidf_matrix.argsort(axis=1)
    sort_indices = np.flip(sort_indices, axis=1)  # to get descending order
    terms = np.array(tfidf.get_feature_names_out())
    return terms[sort_indices[:, :n]].tolist()


def _get_character_keywords(graph: ig.Graph, n: int = 20) -> ig.Graph:
    """Get character keywords for single graph"""
    graph.vs[f"top_{n}_distinctive_words"] = _get_distinctive_words(
        graph.vs["tokens"], n)
    return graph


def _compute_community_structure(graph: ig.Graph, use_weights: bool = False) -> ig.Graph:
    weights = "weight" if use_weights else None
    graph["communities"] = graph.community_walktrap(weights).as_clustering()
    graph.vs["community"] = graph["communities"].membership
    return graph


def _get_community_keywords(graph: ig.Graph, n: int = 20) -> ig.Graph:
    keyed_tokens: KeyedTokens = defaultdict(list)
    for tokens,community in zip(graph.vs["tokens"], graph.vs["community"]):
        keyed_tokens[community] += tokens
    keywords = {comm:kws for comm,kws in zip(keyed_tokens, _get_distinctive_words(keyed_tokens.values(), n))}
    graph["community_keywords"] = keywords
    return graph


def view_characters(corpus: GraphCorpus, play: str, attribute: str, by: str = "word_count", n: int = 5) -> None:
    """Print human-readable view of characters in a given play"""
    chars = corpus[play].vs
    n = len(chars) if n > len(chars) else n
    to_print = [(name,attr,by) for name,attr,by in zip(chars["name"], chars[attribute], chars[by])]
    to_print = sorted(to_print, key=lambda x: x[2], reverse=True)[:n]
    print(f"Showing {attribute.replace('_',' ')} for top {n} characters in {corpus[play]['title']} by {by.replace('_',' ')}\n")
    for name,attr,_ in to_print:
        if isinstance(attr, list):
            attr = textwrap.shorten(", ".join(attr), 68)
        print(f"{name:<25}: {attr}")

def view_community_keywords(corpus: GraphCorpus, play: str) -> None:
    """See most distinctive words per community in the passed play"""
    print(f"Showing most distinctive words by community in {corpus[play]['title']}\n")
    for key,val in corpus["harpur"]["community_keywords"].items():
        print(f"{key:<9}: {', '.join(val)[:90]}...")