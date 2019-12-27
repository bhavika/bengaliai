import os
import wandb
import fastai
from fastai.vision import *
from bengaliai.config import *
from bengaliai.data import get_train
import matplotlib

print(fastai.__version__)

train_df, labels = get_train()

fold = 0

data = (
    ImageList.from_df(
        train_df,
        path=DATA_DIR,
        folder="train",
        suffix=".png",
        cols="image_id",
        convert_mode="L",
    )
    .split_by_idx(
        range(fold * len(train_df) // folds, (fold + 1) * len(train_df) // folds)
    )
    .label_from_df(cols=label_names)
    .transform(
        get_transforms(do_flip=False, max_warp=0.1), size=size, padding_mode="zeros"
    )
    .databunch(bs=batchsize)
).normalize(stats)


data.show_batch()

if __name__ == "__main__":
    # do stuff
    print("Wheee!")
