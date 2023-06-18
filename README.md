# Romanticism and the Contingent Self

Code supporting the analysis in my in-preparation monograph *Romanticism and the Contingent Self*. The main functionality is packaged up in the `romanticself` Python package. Jupyter Notebooks in the home directory show how I generated the tables and figures for each chapter.

## Getting Started

The code in this repository is not entirely reproducible, because some of the data is copyright protected an cannot be published here. However, the actual scripts should run on any machine with Python 3 installed, so long as you install the dependencies:

```{python}
pip3 install nltk
pip3 install bs4
```

## Guide to the repository

The repository is divided into four main parts:

* [data](data): The data directory. Only some of the data can be posted here for copyright reasons. (See below.)
* [figures](figures): Rendered versions of figures used in the book
* [romanticself](romanticself): A Python package bundling the classes used to ingest data for the analysis
* [scripts](scripts): Some  scripts used to run larger analyses on remote machines

In the main directory, you will find one Jupyter Notebook for each chapter of the book. Each notebook displays how I applied the code in the repo to produce the analysis in the corresponding chapter. As mentioned above, this analysis will not be fully reproducible on your machine due to copyright issues in the data.

* [Chapter 1](Chapter%201%20-%20Must%20I%20Exist.ipynb): Analysis of the JSTOR Corpus to establish how 'selfhood' is discussed in contemporary Romantic scholarship
* Chapter 2 does not include any digital analysis.
* [Chapter 3](Chapter%203%20-%20Fiction.ipynb): Analysis of the novel corpus to explore the portrayal of self in the *Bildungsroman* and the 'network novel'
* [Chapter 4](Chapter%204%20-%20Lyric.ipynb): Analysis of the sonnet corpus to explore how genre, narrative and deixis contribute to the portrayal of self in Romantic lyric poetry
* [Chapter 5](Chapter%205%20-%20Drama.ipynb): Network analysis of two drama corpora to establish how characterisation and setting work together to undo the self in Romantic tragedy
* [Chapter 6](Chapter%206%20-%20Life.ipynb): Sentiment analysis of the biography corpus to explore how Thomas Moore and other Romantic biographers construct the selfhood of their subject

## Licences

The code is licensed under the MIT license in the [LICENSE] file.

Some files in [data](data) have other licences. If the licence is 'Gutenberg', then the text of the licence is included in the file. The [CC0](https://creativecommons.org/choose/zero/) licence means the file is in the public domain.

### In the [Novel corpus](data/novel-corpus/)

In addition to the listed files, the analysis in *Romanticism and the Contingent Self* uses the text of five novels that are in copyright.

* [charlotte-temple.txt](data/novel-corpus/charlotte-temple.txt): Gutenberg
* [the-coquette.txt](data/novel-corpus/the-coquette.txt): Gutenberg
* [hobomok.txt](data/novel-corpus/hobomok.txt): [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0)
* [arthur-mervyn.txt](data/novel-corpus/arthur-mervyn.txt): Gutenberg
* [last-of-the-mohicans.txt](data/novel-corpus/last-of-the-mohicans.txt): Gutenberg
* [banished-man.txt](data/novel-corpus/banished-man.txt): [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0)
* [old-manor-house.txt](data/novel-corpus/old-manor-house.txt): [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0)
* [evelina.txt](data/novel-corpus/evelina.txt): Gutenberg
* [camilla.txt](data/novel-corpus/camilla.txt): Gutenberg
* [the-recess.txt](data/novel-corpus/the-recess.txt): CC0
* [maria.txt](data/novel-corpus/maria.txt): Gutenberg
* [julia.txt](data/novel-corpus/julia.txt): CC0
* [simple-story.txt](data/novel-corpus/simple-story.txt): Gutenberg
* [caleb-williams.txt](data/novel-corpus/caleb-williams.txt): Gutenberg
* [anna-st-ives.txt](data/novel-corpus/anna-st-ives.txt): Gutenberg
* [henry.txt](data/novel-corpus/henry.txt): CC0
* [emma-courtney.txt](data/novel-corpus/emma-courtney.txt): Gutenberg
* [belinda.txt](data/novel-corpus/belinda.txt): Gutenberg
* [adeline-mowbray.txt](data/novel-corpus/adeline-mowbray.txt): Gutenberg
* [wild-irish-girl.txt](data/novel-corpus/wild-irish-girl.txt): Gutenberg
* [emma.txt](data/novel-corpus/emma.txt): Gutenberg
* [persuasion.txt](data/novel-corpus/persuasion.txt): Gutenberg
* [self-control.txt](data/novel-corpus/self-control.txt): Gutenberg
* [vivian.txt](data/novel-corpus/vivian.txt): Gutenberg
* [heroine.txt](data/novel-corpus/heroine.txt): Gutenberg
* [waverley.txt](data/novel-corpus/waverley.txt): Gutenberg
* [ivanhoe.txt](data/novel-corpus/ivanhoe.txt): Gutenberg
* [marriage.txt](data/novel-corpus/marriage.txt): Gutenberg
* [annals-of-the-parish.txt](data/novel-corpus/annals-of-the-parish.txt): Gutenberg
* [udolpho.txt](data/novel-corpus/udolpho.txt): Gutenberg
* [melmoth.txt](data/novel-corpus/melmoth.txt): Gutenberg
* [monk.txt](data/novel-corpus/monk.txt): Gutenberg
* [frankenstein.txt](data/novel-corpus/frankenstein.txt): Gutenberg
* [cottagers-of-glenburnie.txt](data/novel-corpus/cottagers-of-glenburnie.txt): [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0)
* [justified-sinner.txt](data/novel-corpus/justified-sinner.txt): Gutenberg

### In the [Biography corpus](data/biography-corpus/)

* [galt.xml](data/biography-corpus/galt.xml): [CC-BY-NC-SA 3.0](http://creativecommons.org/licenses/by-nc-sa/3.0/)
* [lockhart.xml](data/biography-corpus/lockhart.xml): [CC-BY-NC-SA 3.0](http://creativecommons.org/licenses/by-nc-sa/3.0/)
* [medwin.xml](data/biography-corpus/medwin.xml): [CC-BY-NC-SA 3.0](http://creativecommons.org/licenses/by-nc-sa/3.0/)
* [moore.xml](data/biography-corpus/moore.xml): [CC-BY-NC-SA 3.0](http://creativecommons.org/licenses/by-nc-sa/3.0/)
* [gaskell-1.txt](data/biography-corpus/gaskell-1.txt): Gutenberg
* [gaskell-2.txt](data/biography-corpus/gaskell-2.txt): Gutenberg
* [southey.txt](data/biography-corpus/southey.txt): Gutenberg

### In the Sonnet corpus

Unfortunately the entire sonnet corpus is in copyright, and cannot be shared in this repository.

### In the [Drama corpus](data/drama-networks/)

This corpus only contains facts about the network structure of the plays, and is not subject to copyright.

[data]: data