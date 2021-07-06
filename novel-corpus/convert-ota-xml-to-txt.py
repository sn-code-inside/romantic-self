# Looks in current directory, converts every TEI xml file to a txt of the body text

import os

import bs4 as bs

xml_files = [file for file in os.listdir() if file.endswith(".xml")]

print(f"Files found: {', '.join(xml_files)}")

for xml_path in xml_files:
    with open(xml_path, mode="rt") as file:
        soup = bs.BeautifulSoup(file, features="html.parser")
    text = soup.text
    text = text.replace("ſ", "s")
    txt_path = xml_path[:-4] + ".txt"
    with open(txt_path, mode="wt") as file:
        file.write(text)
    
    print(f"Text from {xml_path} written to {txt_path}")