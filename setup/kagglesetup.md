Kaggle API interface is installed in `setup4.sh`.

Copy the contents of `C:\Users\chami\chami_folder\tufts\credentials\kaggle.json` from my desktop to `~/.kaggle/kaggle.json`.

To set up the Kaggle token, I followed [this](https://github.com/Kaggle/kaggle-api#api-credentials).

Then run: 

```bash
chmod 600 ~/.kaggle/kaggle.json
```

Then open a screen for Kaggle: 

```bash
screen -S kaggle
```

Then run the download (as per [here](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)):

```bash
kaggle competitions download -c imagenet-object-localization-challenge
```
