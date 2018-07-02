# News Comments Upvotes

## Installation

`pip install git+https://github.com/jfilter/text-classification-keras#egg=keras_text`

## Setup

Create a `config.py` with the following values

```python
path_data = '/Users/filter/data/pol_comments_selection.csv'
path_embedding = '/Users/filter/data/guardian-twokenized-lower-50.vec'
path_for_proc_data = 'imdb_proc_data.bin'
```

## Usage

1.  Build (preprocess) the dataset
    `python run build`

2.  Train
    `python run train`
