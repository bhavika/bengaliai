import os

HEIGHT = 137
WIDTH = 236
SIZE = 128
stats = (0.0692, 0.2051)

DATA_DIR = '../data'
batchsize = 128

TRAIN = [os.path.join(DATA_DIR, 'bengaliai-cv19', 'train_image_data_0.parquet'),
         os.path.join(DATA_DIR, 'bengaliai-cv19', 'train_image_data_1.parquet'),
         os.path.join(DATA_DIR, 'bengaliai-cv19', 'train_image_data_2.parquet'),
         os.path.join(DATA_DIR, 'bengaliai-cv19', 'train_image_data_3.parquet')]

TEST = [os.path.join(DATA_DIR, 'bengaliai-cv19', 'test_image_data_0.parquet'),
        os.path.join(DATA_DIR, 'bengaliai-cv19', 'test_image_data_1.parquet'),
        os.path.join(DATA_DIR, 'bengaliai-cv19', 'test_image_data_2.parquet'),
        os.path.join(DATA_DIR, 'bengaliai-cv19', 'test_image_data_3.parquet')]


train_df_pth = os.path.join(DATA_DIR, 'bengaliai-cv19', 'train.csv')
test_df_pth = os.path.join(DATA_DIR, 'bengaliai-cv19', 'test.csv')