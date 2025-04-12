# Import libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
import shutil
import re
from PIL import Image

### Maak splits vergelijkbaar met origineel artikel
# Exacte seed is niet gegeven waardoor geen exacte reproductie mogelijk is. 
# De code hieronder maakt vergelijkbare splits op basis van de beschikbare informatie in het artikel.

data_dir          = os.getcwd() + "/dataset/" #dataset directory
lines_dir         = data_dir + "lines/"
splits_csv        = pd.read_csv(data_dir + 'splits.csv', sep= ";")

save_splits       = True
create_subsets    = [10,40,100]

# Groepeer de data op 'form_id' en tel het aantal regels per formulier
splits_csv['folder_id'] = [re.split(pattern = "-", string = i)[0] for i in splits_csv.form_id]
splits_csv['path'] = [lines_dir + 
                      splits_csv.folder_id[i] + "/" + 
                      splits_csv.form_id[i].rstrip("-")   + "/" +
                      splits_csv.form_id_line_id[i] + ".png" for i in range(0, len(splits_csv))]
form_counts = splits_csv.groupby('form_id').size()

# Aantal regels per set (uit artikel)
train_lines = 6161
validation_lines = 966
test_lines = 2915

# Aantal formulieren per set (uit artikel)
train_form_count = 747
validation_form_count = 115
test_form_count = 336

# Randomiseer de form_id's
np.random.seed(42)  # Voor reproduceerbaarheid (in artikel geen seed aangegeven)
form_ids = form_counts.index.to_numpy()
np.random.shuffle(form_ids)

# Start de telling van regels en formulieren voor elke set
train_forms = []
validation_forms = []
test_forms = []

train_counter = 0
validation_counter = 0
test_counter = 0

# Formulieren toevoegen aan de juiste set totdat we de gewenste aantallen bereiken
for form_id in form_ids:
    form_size = form_counts[form_id]

    # Voeg form_id toe aan de train set als we daar ruimte voor hebben
    if len(train_forms) < train_form_count and train_counter + form_size <= train_lines:
        train_forms.append(form_id)
        train_counter += form_size
    # Voeg form_id toe aan de validation set als we daar ruimte voor hebben
    elif len(validation_forms) < validation_form_count and validation_counter + form_size <= validation_lines:
        validation_forms.append(form_id)
        validation_counter += form_size
    # Voeg form_id toe aan de test set als we daar ruimte voor hebben
    elif len(test_forms) < test_form_count and test_counter + form_size <= test_lines:
        test_forms.append(form_id)
        test_counter += form_size

# Maak de uiteindelijke subsets
splits_csv['split'] = 'excluded'
splits_csv.loc[splits_csv['form_id'].isin(train_forms), 'split'] = 'train'
splits_csv.loc[splits_csv['form_id'].isin(validation_forms), 'split'] = 'val'
splits_csv.loc[splits_csv['form_id'].isin(test_forms), 'split'] = 'test'
splits_csv = splits_csv[splits_csv.split != 'excluded']

# Controleer of de aantallen correct zijn
print(f"Train set: {len(splits_csv[splits_csv.split == 'train'])} regels, {len(train_forms)} formulieren.")
print(f"Validation set: {len(splits_csv[splits_csv.split == 'val'])} regels, {len(validation_forms)} formulieren.")
print(f"Test set: {len(splits_csv[splits_csv.split == 'test'])} regels, {len(test_forms)} formulieren.")

# Sla de data op in nieuwe CSV-bestanden
split_dir = data_dir + "splits/"
os.makedirs(split_dir, exist_ok = True)
splits = dict()
for split in splits_csv.split.unique():
    splits_csv[splits_csv.split == split].to_csv(split_dir + split + '_data.csv', index=False)
    splits[split] = splits_csv[splits_csv.split == split]

### Verplaatsen van images naar de juiste mapjes op basis van splits
lines_paths   = glob.glob(f"{lines_dir}/**/*.png", recursive=True) #alle lines

# Kopïeer lines naar split folders
labels_df = pd.read_csv(data_dir + "label_text_lines.csv", sep = ";")

for split in splits.keys():
    os.makedirs(split_dir + split, exist_ok=True)
    paths = []
    labels = []

    for line_path, form_id_line_id in zip(splits[split].path, splits[split].form_id_line_id):
        with Image.open(line_path) as img:
            img_resized = img.resize((int(128 * img.width / img.height), 128), Image.LANCZOS)
            img_resized.save(split_dir + split + "/" + form_id_line_id + ".png")
        paths.append(split_dir + split + "/" + form_id_line_id + ".png")
        labels.append(labels_df.loc[labels_df['filename'] == form_id_line_id, 'trancription'].values[0])
    
    full_data_dir = split_dir + "full_dataset/"
    os.makedirs(full_data_dir, exist_ok=True)

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