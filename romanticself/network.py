"""Functions for importing and analysing network data for Chapter 5"""

from collections import defaultdict, namedtuple
from itertools import chain
import re
from typing import Iterable, TypeAlias
import os
import json
import csv
import igraph as ig
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

GraphCorpus: TypeAlias = dict[str, ig.Graph]


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
        graph = _update_vert_attrs_from_cast(graph, raw_dir, play)
        # graph = _collect_tokens_by_character(graph)
        corpus[play] = graph
    if cache_corpus:
        _save_corpus_as_gml(data_dir, corpus)
    return corpus


def _save_corpus_as_gml(data_dir: str, corpus: GraphCorpus) -> None:
    for graph in corpus.values():
        _graph = graph.copy()
        del _graph.es["tokens"]  # cannot be published to github
        _graph.write_graphml(_get_gml_filename(data_dir, graph["play"]))


def _import_from_cache(data_dir: str, manifest: dict) -> GraphCorpus:
    return {play: ig.Graph.Read_GraphML(_get_gml_filename(data_dir, play)) for play in manifest}


def _get_gml_filename(data_dir: str, play: str) -> str:
    return os.path.join(data_dir, f"{play}.gml")


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
    word_rgx = re.compile(r"\w")
    edges = defaultdict(list)
    with open(os.path.join(raw_dir, f"{play}EdgeList.csv"), "rt", encoding="utf8") as edgelist:
        reader = csv.reader(edgelist)
        _ = next(reader)
        lang = "french" if play == "iphigenie" else "english"
        for src, tar, text in reader:
            edges[src, tar] += (token for token in nltk.word_tokenize(text.lower(), language=lang)
                                if word_rgx.match(token))
    return [ImportEdge(src, tar, tokens, len(tokens)) for (src, tar), tokens in edges.items()]


def _update_vert_attrs_from_cast(graph: ig.Graph, raw_dir: str, play: str) -> ig.Graph:
    with open(os.path.join(raw_dir, f"{play}Cast.csv"), "rt", encoding="utf8") as vertexlist:
        vertices = csv.reader(vertexlist)
        # columns after 'code' contain node attributes
        vert_attrs = next(vertices)[1:]
        for code, *attrs in vertices:
            vertex_idx = graph.vs["code"].index(code)
            for key, val in zip(vert_attrs, attrs):
                graph.vs[vertex_idx][key] = val
    return graph

def _collect_tokens_by_character(graph: ig.Graph) -> ig.Graph:
    # TODO: Implement this to get character-level word usage data from the Excel spreadsheets
    # This cannot be done from the graph itself, because each character's speeches to all other
    # characters are concatenated
    NotImplemented

def get_character_keywords(corpus: GraphCorpus, n: int = 20) -> list:
    """Compute distinctive words for each character in the corpus using tf-idf"""
    out = list()
    for graph in corpus.values():
        tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
        tfidf_matrix = tfidf.fit_transform(graph.vs["tokens"])
        out.append(tfidf)
    return out

def _get_community_keywords(corpus: GraphCorpus, n: int = 20) -> dict:
    pass
