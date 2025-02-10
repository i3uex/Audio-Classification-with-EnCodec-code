import os
import shutil

out_dir = "classes"
base_folder = "datasets/ESC-50-master"
with open(f"{base_folder}/meta/esc50.csv", "r") as f:
    lines = [x.strip().split(",") for x in f.readlines()[1:]]

for filename, fold, target, category, esc10, src_file, take in lines:
    dest_fpath = f"{base_folder}/{out_dir}/{category}/{filename}"
    os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
    shutil.copy(f"{base_folder}/audio/{filename}", dest_fpath)
