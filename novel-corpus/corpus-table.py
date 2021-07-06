# Converts the corpus manifest into a csv for adding to book

import json

with open("manifest.json", "rt") as file:
    manifest = json.load(file)

csv = "Title,Author,Year,Nation,Gothic,Network,Source,Available Online\n"

def src(url):
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

def avail(licence):
    if licence == "Restrictive":
        return "No"
    else:
        return "Yes"

def bool2str(bool_val):
    if bool_val:
        return "Yes"
    else:
        return "No"

sorted_manifest = sorted(manifest.values(), key=lambda x: x["year"])

novels = []
for novel in sorted_manifest:
    # Write to csv
    csv += novel["short_title"] + ","
    csv += novel["author"] + ","
    csv += str(novel["year"]) + ","
    csv += novel["nation"] + ","
    csv += bool2str(novel["gothic"]) + ","
    csv += bool2str(novel["network"]) + ","
    if isinstance(novel["source"], list):
        csv += src(novel["source"][0]) + ","
    else:
        csv += src(novel["source"]) + ","
    csv += avail(novel["licence"])
    csv += "\n"

with open("corpus.csv", "wt") as file:
    file.write(csv)