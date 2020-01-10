from fastai import train
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from joblib import Parallel, delayed
from bengaliai.config import (
    HEIGHT,
    WIDTH,
    SIZE,
    stats,
    train_df_pth,
    TRAIN,
    TEST,
    batchsize,
    DATA_DIR,
)
import numpy as np
import cv2
import os
import tqdm
from skimage.io import imsave


class BengaliAIDataset(Dataset):
    def __init__(self, fname):
        self.df = pd.read_parquet(fname)

    def bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def crop_resize(self, img0, size=SIZE, pad=16):
        # crop a box around pixels large than the threshold
        # some images contain line at the sides
        ymin, ymax, xmin, xmax = self.bbox(img0[5:-5, 5:-5] > 80)
        # cropping may cut too much, so we need to add it back
        xmin = xmin - 13 if (xmin > 13) else 0
        ymin = ymin - 10 if (ymin > 10) else 0
        xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
        ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
        img = img0[ymin:ymax, xmin:xmax]
        # remove lo intensity pixels as noise
        img[img < 28] = 0
        lx, ly = xmax - xmin, ymax - ymin
        l = max(lx, ly) + pad
        # make sure that the aspect ratio is kept in rescaling
        img = np.pad(img, [((l - ly) // 2,), ((l - lx) // 2,)], mode="constant")
        return cv2.resize(img, (size, size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.df.iloc[idx, 0]
        data = 255 - self.df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
        img = (data[idx] * (255.0 / data[idx].max())).astype(np.uint8)
        img = self.crop_resize(img)
        return img, name


def create_train_images():
    def write_img(item, ds_id):
        img, name = item[0], item[1]
        try:
            if not os.path.exists(os.path.join(DATA_DIR, "train", f"{name}_{ds_id}.png")):
                cv2.imwrite(os.path.join(DATA_DIR, "train", f"{name}_{ds_id}.png"), img)
        except Exception as e:
            print(f"Could not save image {name} due to {e}")

    for ds_id, f in enumerate(TRAIN):
        ds = BengaliAIDataset(f)
        Parallel(n_jobs=6)(delayed(write_img)(item, ds_id) for _, item in tqdm.tqdm(enumerate(ds, 0)))


def get_train():
    df = pd.read_csv(train_df_pth)
    nunique = list(df.nunique())[1:-1]
    return df, nunique


create_train_images()