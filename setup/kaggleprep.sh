chmod 600 ~/.kaggle/kaggle.json

# Taken from here:
# https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data
kaggle competitions download -c imagenet-object-localization-challenge
mkdir ImageNet
time unzip imagenet-object-localization-challenge.zip -d ImageNet >& /dev/null
mv ImageNet Net2Net/data
rm imagenet-object-localization-challenge.zip

# Taken from here: https://github.com/facebookresearch/kill-the-bits/issues/5,
# https://discuss.pytorch.org/t/issues-with-dataloader-for-imagenet-should-i-use-datasets-imagefolder-or-datasets-imagenet/115742/7
# Related error discussed here: 
# https://stackoverflow.com/questions/69199273/torchvision-imagefolder-could-not-find-any-class-folder
time bash valprep.sh