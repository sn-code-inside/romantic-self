# Contingent Selves: Romanticism and the Challenge of Representation

Code supporting the analysis in my in-preparation monograph *Contingent Selves: Romanticism and the Challenge of Representation*. The main functionality is packaged up in the `romanticself` Python package. Jupyter Notebooks in the home directory show how I generated the tables and figures for each chapter.

## Getting Started

The code in this repository is not entirely reproducible, because some of the data is copyright protected an cannot be published here. However, the actual scripts should run on any machine with Python 3 installed, so long as you install the dependencies:

```{python}
pip3 install nltk
pip3 install bs4
```

## Guide to the repository

The repository is divided into four main parts:

* `data`: The data directory. Only some of the data can be posted here for copyright reasons.
* `figures`: Rendered versions of figures used in the books
* `romanticself`: A Python package bundling the classes used to perform the analysis
* `scripts`: Some  scripts used to run larger analyses on remote machines

In the main directory, you will find one Jupyter Notebook for each chapter of the book. Each notebook displays how I applied the code in the repo to produce the analysis in the corresponding chapter. As mentioned above, this analysis will not be reproducible on your machine due to copyright issues in the data.
