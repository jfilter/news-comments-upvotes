# News Comments Upvotes

## Installation

Please use this very commit. It wouldn't work with the latest changes.

```bash
pip install tensorflow
pip install git+https://github.com/jfilter/text-classification-keras@8d5b4efb500be529a8454bab88c182953a64c995#egg=keras_text
```

## Config

Create a `config.py` with the following variables (and your values)

```python
path_data = '/Users/filter/data/pol_comments_selection.csv'
path_embedding = '/Users/filter/data/guardian-twokenized-lower-50.vec'
path_for_proc_data = 'imdb_proc_data.bin'
base_experiment_folder 'somefolder'
```

## Usage

1.  Build (preprocess) the dataset
    `python run.py build`

2.  Train e.g. a CNN
    `python run.py traincnn`
