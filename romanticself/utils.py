from romanticself.corpus import EagerCorpus

def import_scheme_map(path: str) -> dict:
    """Import rhyme scheme map from csv file"""

    scheme_map = {}

    with open(path, "rt") as scheme_file:
        for line in scheme_file.readlines()[1:]:
            scheme, code = line.split(",")
            scheme_map[scheme] = code.strip()

    return scheme_map