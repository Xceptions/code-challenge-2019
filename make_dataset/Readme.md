# Make Dataset task

This script will separate the dataset into two
files; train.csv and test.csv and save to
local disk

```
Usage: dataset.py [OPTIONS]

  I use pandas for this operation.
  Split the dataset using numpy in the ratio
  of 80 for the train set and 20 for the
  test set then save them to one file

  Returns ------- train.csv and test.csv

Options:
  --name TEXT
  --url TEXT
  --out-dir TEXT
  --help          Show this message and exit.
```
