Kaggle API interface is installed in `setup4.sh`.

Copy the contents of `C:\Users\chami\chami_folder\tufts\credentials\kaggle.json` from my desktop to `~/.kaggle/kaggle.json`.

To set up the Kaggle token, I followed [this](https://github.com/Kaggle/kaggle-api#api-credentials).

Then run: 

```bash
chmod 600 ~/.kaggle/kaggle.json
```

Then open a screen for Kaggle (the download takes about 30 minutes on c240g5, the extraction also takes about 30 minutes): 

```bash
screen -S kaggle
```

Then run the download (as per [here](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)) using this script:

```bash
kaggle competitions download -c imagenet-object-localization-challenge
mkdir ImageNet
time unzip imagenet-object-localization-challenge.zip -d ImageNet >& /dev/null
mv ImageNet Net2Net/data
rm imagenet-object-localization-challenge.zip
```

