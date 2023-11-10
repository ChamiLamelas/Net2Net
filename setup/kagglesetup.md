Kaggle API interface is installed in `setup4.sh`.

Copy the contents of `C:\Users\chami\chami_folder\tufts\credentials\kaggle.json` from my desktop to `~/.kaggle/kaggle.json`.

To set up the Kaggle token, I followed [this](https://github.com/Kaggle/kaggle-api#api-credentials).

Then open a screen for Kaggle (the download takes about 30 minutes on c240g5, the extraction also takes about 30 minutes): 

```bash
screen -S kaggle
```

Then in `setup/` run:

```bash
bash kaggleprep.sh
```
