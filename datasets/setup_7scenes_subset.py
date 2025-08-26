#!/usr/bin/env python3

import argparse
import os
import warnings

import dataset_util as dutil
import numpy as np
import torch
from joblib import Parallel, delayed
from skimage import io



# name of the folder where we download the original 7scenes dataset to
# we restructure the dataset by creating symbolic links to that folder
src_folder = '7scenes'
focal_length = 525.0

# focal length of the depth sensor (source: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
d_focal_length = 585.0

# RGB image dimensions
img_w = 640
img_h = 480


# sub sampling factor of eye coordinate tensor
nn_subsampling = 8

# transformation from depth sensor to RGB sensor
# calibration according to https://projet.liris.cnrs.fr/voir/activities-dataset/kinect-calibration.html
d_to_rgb = np.array([
    [9.9996518012567637e-01, 2.6765126468950343e-03, -7.9041012313000904e-03, -2.5558943178152542e-02],
    [-2.7409311281316700e-03, 9.9996302803027592e-01, -8.1504520778013286e-03, 1.0109636268061706e-04],
    [7.8819942130445332e-03, 8.1718328771890631e-03, 9.9993554558014031e-01, 2.0318321729487039e-03],
    [0, 0, 0, 1]
])
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Setup 7scenes dataset')
    parser.add_argument('--dataset_path', type=Path, default='datasets/7scenes',
                        help='Path to the 7scenes dataset')
    parser.add_argument('--dump_path', type=Path, default='7scenes',)
    parser.add_argument("--subset_path", type=Path, default='datasets/7scenes_subset')
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        # os.makedirs(args.dataset_path)
        raise FileNotFoundError(f"Dataset path {args.dataset_path} does not exist. Please download the 7scenes dataset and place it there.")
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)
    if not os.path.exists(args.subset_path):
        raise FileNotFoundError(f"Subset path {args.subset_path} does not exist. Please create it before running the script.")
    

    subset_folders = sorted([f for f in args.subset_path.iterdir() if f.is_dir()])

    for scene_folder in subset_folders:
        scene_name = scene_folder.name
        print(f"Processing scene: {scene_name}")
        os.makedirs(args.dump_path / scene_name, exist_ok=True)

        subset_name = scene_folder / f"image_names.txt"
        if not os.path.exists(subset_name):
            warnings.warn(f"Subset file {subset_name} does not exist. Skipping scene {scene_name}.")
            continue
        subsetnames = np.loadtxt(subset_name, dtype=str)
        # print(subsetnames[0][0])
        subsetnames = [ s[2:-2] for s in subsetnames]  # Remove quotes from the names
        # subsetnames[:,6] = "/"
        print(subsetnames[6])
        # for _ in range(len(subsetnames)):
        #     subsetnames[_]=subsetnames[_].replace("-frame", "/frame")
        print(f"Subset names: {subsetnames}")
        cnt = 0
        for _ in range(len(subsetnames)):
            img_file = args.dataset_path  / scene_name / subsetnames[_].replace("-frame", "/frame")
            link_path = args.dump_path / scene_name / (f"{cnt:05d}_"+subsetnames[_])
            cnt += 1
            import shutil
            # if not os.path.exists(link_path):
            #     if not os.path.exists(img_file):
            #         warnings.warn(f"Image file {img_file} does not exist. Skipping.")
            #         continue
                # Create a symbolic link to the image file in the dump path
                # os.symlink(img_file, link_path)
                # shutil.copy(img_file, link_path)
            os.symlink(img_file, link_path)

