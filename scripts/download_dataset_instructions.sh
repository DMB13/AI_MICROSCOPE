#!/bin/sh
# Instructions to acquire the DiBaS dataset or an equivalent labeled dataset.
# This script does not download automatically because dataset licensing may
# require manual agreement. Follow these steps:
# 1. Visit the DiBaS dataset page (Zielinski et al.) or your dataset provider.
# 2. Request or download the dataset and extract it locally.
# 3. Place the dataset root at `data/raw`.
# 4. Expected layout (preferred):
#    data/raw/<class_name>/*.jpg
#    e.g. data/raw/Escherichia_coli/img1.jpg
# 5. Run verification:
#    PYTHONPATH=. python3 scripts/dataset_verify.py --dataset data/raw

echo "See the comments in this script for manual download steps and verification." > /dev/null
