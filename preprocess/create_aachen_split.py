# Import libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
import shutil
import re
from PIL import Image

### Maak splits op basis van artikel 

data_dir          = os.getcwd() + "/dataset/" #dataset directory
lines_dir         = data_dir + "lines/"
splits_csv        = pd.read_csv(data_dir + 'splits.csv', sep= ";")

create_subsets    = [10,40,100]

# Inlezen splits
splits = dict()
for file, split in zip(["te.lst", "tr.lst", "va.lst"], ["test", "train", "val"]):
    with open(data_dir + file, 'r', encoding='utf-8') as f:
        splits[split] = [line.strip() for line in f.readlines()]

# # Controleer of de aantallen correct zijn
for split in splits.keys():
    print(f"{split} set heeft {len(splits[split])} items.")

# Sla de data op in nieuwe CSV-bestanden
split_dir = data_dir + "splits/"
os.makedirs(split_dir, exist_ok = True)

# ### Verplaatsen van images naar de juiste mapjes op basis van splits
lines_paths   = glob.glob(f"{lines_dir}/**/*.png", recursive=True) #alle lines

# Kopïeer lines naar split folders
labels_df = pd.read_csv(data_dir + "label_text_lines.csv", sep = ";")
full_data_dir = split_dir + "full_dataset/"
os.makedirs(full_data_dir, exist_ok=True)

for split in splits.keys():
    os.makedirs(split_dir + split, exist_ok=True)
    paths  = []
    labels = []

    for line in splits[split]:
        filename_parts = line.split("-")
        line_path = lines_dir + filename_parts[0] + "/" + filename_parts[0] + "-" + filename_parts[1] + "/" + line + ".png"
        with Image.open(line_path) as img:
            img_resized = img.resize((int(128 * img.width / img.height), 128), Image.LANCZOS)
            img_resized.save(split_dir + split + "/" + line + ".png")
        paths.append("/dataset/splits/"+ split + "/" + line + ".png")
        labels.append(labels_df.loc[labels_df['filename'] == line, 'trancription'].values[0])

    with open(full_data_dir + split + '_labels.txt', 'w', encoding='utf-8') as f1, \
         open(full_data_dir + split + '_ids.txt', 'w', encoding='utf-8') as f2, \
             open(full_data_dir + split + '_eval.txt', 'w', encoding='utf-8') as f3, \
                 open(full_data_dir + split + '.txt', 'w', encoding='utf-8') as f4:
         for path, label in zip(paths, labels):
             f1.write(f"{path} {label}\n")
             f2.write(f"{path}\n")
             f3.write(f"{path}\n")
             f4.write(f"{path} {' '.join(list(str(label)))}\n")


def create_subset(files, split_dir, subset):
    sub_dir = split_dir + "subset_" + str(subset)
    os.makedirs(sub_dir, exist_ok=True)
    for filename in files:
        with open(os.path.join(split_dir + "full_dataset/", filename), 'r', encoding='utf-8') as orig_file, \
             open(os.path.join(sub_dir, filename), 'w', encoding='utf-8') as subsetfile:
            lines = orig_file.readlines()
            for line in lines[:subset]:
                subsetfile.write(line)

files = os.listdir(full_data_dir)
files = [f for f in files if ".txt" in f and ("test" in f or "val" in f or "train" in f)]
for subset in create_subsets:
    create_subset(files, split_dir, subset=subset)

# Move syms.txt naar split folder
shutil.copy(data_dir + "syms.txt", split_dir + "syms.txt")