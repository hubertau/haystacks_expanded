# Checkworthiness

This repository contains the code for the project [Developing New Open Source Automated Fact Checking Tools](https://www.oii.ox.ac.uk/research/projects/developing-new-open-source-automated-fact-checking-tools/).

This package allows the user to specify a search query for TikTok, download data, train or run a model on the data, and output a result.

# Quick Start

Clone the repository, and install the package locally:

`pip install -e haystacks_expanded`

Please create a file queries.json where you may enter the parameters of download queries. For example:

```json
{
    "health": {
        "text": "covid AND vaccine",
        "period": 1
    },
    "health7": {
        "text": "covid vaccine hoax leak",
        "period": 7
    },
    "health90": {
        "text": "covid vaccine hoax leak",
        "period": 90
    },
    "steroids90": {
        "text": "#tren bodybuilding workout #skinny #thin #trensetter #notnatty #dianabol #anadrol #peptides #roidtok",
        "period": 90
    }
}
```

An EnsembleData (https://ensembledata.com/) API key is also required.

On an initial run, it will prompt you to set a number of directories: where you would like to store files, your API key (which will be kept in your home directory, by default), etc.

To download a query names `health`:

`hay download health`


