import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R

from benchmarks.preprocess_data import convert_opengl_to_opencv
from benchmarks.ransac_eval_pose import eval_pose_acc


def matrix_to_quaternion(matrix):
    R_matrix = matrix[:3, :3]
    r = R.from_matrix(R_matrix)
    q = r.as_quat(scalar_first=True)  # Convert to quaternion (w,x,y,z)
    t = matrix[:3, 3]
    return q, t


def quaternion_to_matrix(
    q: List[float], t: List[float], input_quat_type="wxyz"
) -> List[List[float]]:
    # Convert quaternion to rotation matrix
    # Scipy wants xyzw format, so we need to permute the components if the input is wxyz:
    if input_quat_type == "wxyz":
        r = R.from_quat([q[1], q[2], q[3], q[0]])
    else:
        assert (
            input_quat_type == "xyzw"
        ), f"Unexpected input_quat_type {input_quat_type}"
        r = R.from_quat([q[0], q[1], q[2], q[3]])
    matrix = r.as_matrix()

    # Construct 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = matrix
    transform[:3, 3] = t

    return transform.tolist()


def count_colmap_valid_mask(colmap_path, frame_num):
    valid_mask = []
    gt_poses = []
    for i in range(frame_num):
        if "training" in colmap_path:
            colmap_file = os.path.join(colmap_path, f"{i:06d}_pose.txt")
        else:
            colmap_file = os.path.join(colmap_path, f"{i+1:05d}_pose.txt")
        if not os.path.exists(colmap_file):
            valid_mask.append(0)
        else:
            with open(colmap_file, "r") as f:
                lines = f.readlines()
            if "inf" in lines[0] or "nan" in lines[0]:
                valid_mask.append(0)
            else:
                gt_poses.append(np.loadtxt(colmap_file).reshape(4, 4))
                valid_mask.append(1)
    return np.array(valid_mask, dtype=np.int32).astype(np.bool_), gt_poses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=Path, required=True, help="Path to the file containing poses."
    )
    parser.add_argument(
        "--ba", action="store_true", help="eval the poses after bundle adjustment"
    )
    args = parser.parse_args()
    scene_folders = sorted([f for f in args.dir.iterdir() if f.is_dir()])

    keys = [
        "ate",
        "auc5",
        "r_rra_5",
        "t_rra_5",
    ]
    header_str = "Scene: "
    for key in keys:
        header_str += key + " "
    print(header_str)
    total_scenes = 0
    total_dict = {key: 0.0 for key in keys}
    for scene_folder in scene_folders:
        result_file = scene_folder / "eval_result.json"
        gt_poses = scene_folder / "gt.txt"
        aligned_poses = scene_folder / "pred.txt"

        if not aligned_poses.exists():
            print(f"Results not found for {scene_folder.name}.")
            continue
        eval_result = eval_pose_acc(gt_poses, aligned_poses, save_dir=scene_folder)
        if not result_file.exists():
            print(f"Results not found for {scene_folder.name}.")
            continue

        with open(result_file, "r") as f:
            data = json.load(f)

        out_str = f"{scene_folder.name:25s}: "
        for key in keys:
            if key in data and "rra" in key:
                out_str += f"{key}={data[key]*100:04f}% "
                total_dict[key] += data[key]
            elif key in data:
                out_str += f"{key}={data[key]:04f} "
                total_dict[key] += data[key]
            else:
                out_str += f"{key}=N/A "
        total_scenes += 1
        print(out_str)
    print("Total: ", end="")
    for key in keys:
        if "rra" in key:
            print(f"{key}={total_dict[key]*100/total_scenes:04f}%", end=" ")
        else:
            print(f"{key}={total_dict[key]/total_scenes:04f}", end=" ")
