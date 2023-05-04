from sklearn.model_selection import train_test_split
import shutil
import os
import numpy as np
import random

def make_directories(base_dir, subdirs):
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir))

base_dir = "testingdataset/"
make_directories(base_dir, ["test/banjir", "valid/banjir", "train/banjir"])
make_directories(base_dir, ["test/tidak banjir", "valid/tidak banjir", "train/tidak banjir"])
make_directories(base_dir, ["test/bermasalah", "valid/bermasalah", "train/bermasalah"])

dirs = ["banjir", "tidak banjir", "bermasalah"]

for d in dirs:
    src_dir = os.path.join("testingdataset", "@" + d)
    test_dir = os.path.join("testingdataset", "test", d)
    val_dir = os.path.join("testingdataset", "valid", d)
    train_dir = os.path.join("testingdataset", "train", d)
    
    files = [entry.path for entry in os.scandir(src_dir) if entry.is_file()]
    random.shuffle(files)
    n = len(files) #total files = 425
    train_split = int(0.6 * n) # 425 x 0.6=255
    val_split = int(0.2 * n) #425 x 0.2=85 file
    test_split = n - train_split - val_split #425-255-85=85 files
    
    train_files = files[:train_split]#255
    val_files = files[train_split:train_split+val_split]#255/255+85=85
    test_files = files[train_split+val_split:] #255+85=340, 425-340=85
    
    for file in train_files:
        shutil.copy(file, os.path.join(train_dir, os.path.basename(file)))
    for file in val_files:
        shutil.copy(file, os.path.join(val_dir, os.path.basename(file)))
    for file in test_files:
        shutil.copy(file, os.path.join(test_dir, os.path.basename(file)))