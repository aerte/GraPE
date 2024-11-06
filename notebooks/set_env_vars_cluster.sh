#!/bin/bash

# copy this script to the directory where you want to run the notebooks from, e.g. root of GraPE
# To run it: 
#   chmod +x set_env_vars_cluster.sh
#   source set_env_vars_cluster.sh

# set the base directory. Replace with your own home directory on the cluster
export BASE_R_DIR="/zhome/8c/5/143289/GraPE"

export DATA_SPLITS_PATH="$BASE_R_DIR/env/data_splits.xlsx"
export PKA_DATASET_PATH="$BASE_R_DIR/env/pka_dataset.xlsx"
export FRAGMENTATION_SCHEME_PATH="$BASE_R_DIR/env/MG_plus_reference"
export SAVE_FRAGMENTATION_PATH="$BASE_R_DIR/env/fragmentation_data"
export RAY_TEMP_DIR="$BASE_R_DIR/tmp_ray"
export BOHB_RESULTS_DIR="$BASE_R_DIR/env/bohb_results"

# (Optional) Create directories if they do not exist
# mkdir -p "$SAVE_FRAGMENTATION_PATH"
# mkdir -p "$RAY_TEMP_DIR"
# mkdir -p "$BOHB_RESULTS_DIR"