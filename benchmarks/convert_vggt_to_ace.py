import torch
import numpy as np
from typing import Optional
import shutil
from dataclasses import dataclass
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from benchmarks.preprocess_data import Resolution, Frame, glob_for_frames, get_resolution_from_frames

def matrix_to_quaternion(matrix):
    R_matrix = matrix[:3, :3]
    r = R.from_matrix(R_matrix)
    q = r.as_quat(scalar_first=True)  # Convert to quaternion (w,x,y,z)
    t = matrix[:3, 3]
    return q,t

def convert_vggt_poses_to_ace_poses(data_pattern:str, pose_file: str, intrinsics_file: str, intrinsics_type:str, confidence_depth_file: Optional[str], confidence_point_file: Optional[str], output_file:str, split_json: Optional[str], split_set:str) -> None:
    
    c2ws = np.loadtxt(pose_file).reshape(-1,3,4)
    bottom = np.array([[0, 0, 0, 1]]).reshape(1,1,4)
    c2ws = np.concatenate([c2ws, bottom.repeat(c2ws.shape[0], axis=0)], axis=1)
    w2cs = np.linalg.inv(c2ws)


    qs,ts = [],[]
    for w2c in w2cs:
        q,t = matrix_to_quaternion(w2c)
        qs.append(q)
        ts.append(t)

    dataset_frames = glob_for_frames(data_pattern)
    dataset_frames = sorted(dataset_frames, key=lambda x: x.rgb_path)

    K33 = np.loadtxt(intrinsics_file).reshape(-1,3,3)

    K_h,K_w = K33[0, 1, 2]*2, K33[0, 0, 2]*2
    resolution = get_resolution_from_frames(frames=dataset_frames)

    scale_h, scale_w = resolution.height / K_h, resolution.width / K_w
    K33[:, 0, 2] *= scale_w
    K33[:, 1, 2] *= scale_h
    K33[:, 0, 0] *= scale_w
    K33[:, 1, 1] *= scale_h

    if confidence_depth_file is not None:
        confidence_depth = np.loadtxt(confidence_depth_file).reshape(-1, 1)
    if confidence_point_file is not None:
        confidence_point = np.loadtxt(confidence_point_file).reshape(-1, 1)
    
    print(dataset_frames)

    if split_json is not None:
        with open(split_json, 'r') as f:
            split_data = json.load(f)
        if split_set == "train":
            split_frames = split_data['train_filenames']
        elif split_set == "test":
            split_frames = split_data['test_filenames']
        elif split_set == "all":
            split_frames = split_data['train_filenames'] + split_data['test_filenames']
        else:
            raise ValueError(f"Unknown split set: {split_set}")
        print(split_frames)
        dataset_frames=[f for f in dataset_frames if str(f.rgb_path) in split_frames]

    confidences = []
    if confidence_depth_file is  None and confidence_point_file is  None:
        confidences = [1001 for _ in range(len(dataset_frames))]       
        
    for frame in dataset_frames:
        confidences.append(1001)

    fx = np.median(K33[...,0,0])

    # print(confidences)
    with open(output_file, 'w') as f:
        for frame,q,t,k,c in zip(dataset_frames, qs,ts, K33, confidences):
            print(f"Processing frame: {frame.rgb_path}")
            f.write(f"{frame.rgb_path} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {fx} {c}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="convert vggt poses to ace format")
    parser.add_argument("--pred_poses", type=str, required=True, help="the predicted poses file in vggt format")
    parser.add_argument("--intrinsics", type=str, required=True, help="the intrinsics file in vggt format")
    parser.add_argument("--data_pattern", type=str, required=True, help="the glob pattern for the input data")
    parser.add_argument("--output_file", type=str, required=True, help="the output file in ace format")

    # parser.add_argument("--confidence_depth", type=str, required=False,
    #                     help="the confidence depth file in vggt format")
    # parser.add_argument("--confidence_depth_thres", type=str, required=False,
    #                     help="the confidence depth file in vggt format")
    # parser.add_argument("--confidence_point", type=str, required=False,
    #                     help="the confidence point file in vggt format")
    # parser.add_argument("--")

    parser.add_argument("--split_json", type=str, required=False,
                        help="Path to a JSON file containing splits; if not given, every 8 images")
    parser.add_argument("--split_set", type=str, choices=["train","test","all"], default="all",)

    args = parser.parse_args()

    convert_vggt_poses_to_ace_poses(data_pattern=args.data_pattern,
                                     pose_file=args.pred_poses,
                                     intrinsics_file=args.intrinsics,
                                     intrinsics_type="pinhole",
                                     confidence_depth_file=None,
                                     confidence_point_file=None,
                                     output_file=args.output_file,
                                     split_json=args.split_json,
                                     split_set=args.split_set)