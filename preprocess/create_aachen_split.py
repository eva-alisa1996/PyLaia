# Import libraries
import pandas as pd
import os
import glob
import shutil
from PIL import Image
from typing import Optional

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
             label = ' <space> '.join([' '.join(word) for word in label.strip().split()])
             f4.write(f"{path} {label}\n")


def create_subset(
        files: list, 
        split_dir: str, 
        subset: int):
    
    """
    Genereert een subset van de splits voor tests.

    Args:
        files: list van txt files waarvan een susbet gemaakt moet worden
        split_dir: directory waar de nieuwe split opgeslagen moet worden
        subset: grote van de subset (bijv. 10)
    """

    
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

# Create syms table
def generate_syms_from_labels(
    labels: list,
    output_syms_file: str,
    include_blank: bool = True,
    special_tokens: Optional[list] = None,
):
    """
    Genereert een syms.txt bestand op basis van unieke karakters in het labelbestand.

    Args:
        labels: list met alle labels waaruit de syms.txt gemaakt moet worden
        include_blank: Voeg <ctc> als eerste token toe.
        special_tokens: Extra tokens zoals "<space>" of "<unk>".
    """

    # Zet alle labels om naar een set van unieke symbolen
    symbols = set("".join(labels))
    symbols = sorted(symbols)

    index = 0
    lines = []

    # Voeg <eps> token toe als include_blank True is
    if include_blank:
        lines.append("<ctc> 0")
        index += 1

    # Voeg speciale tokens toe als die zijn opgegeven
    if special_tokens:
        for token in special_tokens:
            lines.append(f"{token} {index}")
            index += 1

    # Voeg de symbolen toe, vervang spaties door <space>
    for s in symbols:
        printable_symbol = "<space>" if s == " " else s
        lines.append(f"{printable_symbol} {index}")
        index += 1

    with open(output_syms_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


syms_table = generate_syms_from_labels(labels = labels_df.trancription, 
                                       output_syms_file = data_dir + "syms.txt")
