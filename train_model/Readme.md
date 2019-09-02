# Train Model Task

This script will fetch the csv file that has been saved on
disk, perform feature engineering, train the model and
save it to disk

```
Usage: train_model.py [OPTIONS]

  train the model and save it to disk

  <!-- Parameters ---------- name: str     name of the csv file on local disk,
  without '.csv' suffix. url: str     remote url of the csv file. out_dir:
  directory where file should be saved to. -->

  Returns ------- Model and Persistence Variables

<!-- Options:
  --name TEXT
  --url TEXT
  --out-dir TEXT
  --help          Show this message and exit. -->
```
