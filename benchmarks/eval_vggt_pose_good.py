import argparse
import numpy as np
import json
import glob
from pathlib import Path
import torch

from benchmarks.ransac_eval_pose import eval_pose_acc
from benchmarks.preprocess_data import convert_nerf_blender_format_to_vggt_format,convert_opencv_to_opengl,convert_opengl_to_opencv
import os
from scipy.spatial.transform import Rotation as R
from typing import List

def matrix_to_quaternion(matrix):
    R_matrix = matrix[:3, :3]
    r = R.from_matrix(R_matrix)
    q = r.as_quat(scalar_first=True)  # Convert to quaternion (w,x,y,z)
    t = matrix[:3, 3]
    return q,t
def quaternion_to_matrix(q: List[float], t: List[float], input_quat_type='wxyz') -> List[List[float]]:
    # Convert quaternion to rotation matrix
    # Scipy wants xyzw format, so we need to permute the components if the input is wxyz:
    if input_quat_type == 'wxyz':
        r = R.from_quat([q[1], q[2], q[3], q[0]])
    else:
        assert input_quat_type == 'xyzw', f'Unexpected input_quat_type {input_quat_type}'
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
            colmap_file = os.path.join(colmap_path,  f"{i:06d}_pose.txt")
        else:
            colmap_file = os.path.join(colmap_path,  f"{i+1:05d}_pose.txt")
        if not os.path.exists(colmap_file):
            valid_mask.append(0)
        else:
            with open(colmap_file, 'r') as f:
                lines = f.readlines()
            if "inf" in lines[0] or "nan" in lines[0]:
                valid_mask.append(0)
            else:
                gt_poses.append(np.loadtxt(colmap_file).reshape(4, 4))
                valid_mask.append(1)
    return np.array(valid_mask, dtype=np.int32).astype(np.bool_), gt_poses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True, help="Path to the file containing poses.")
    parser.add_argument("--ba", action="store_true",  help="eval the poses after bundle adjustment")
    args = parser.parse_args()
    scene_folders = sorted([f for f in args.dir.iterdir() if f.is_dir()])

    keys= ["ate","auc5","r_rra_5","t_rra_5",]
    # keys= ["ate", "auc30","auc15","auc10","auc5","auc2","auc1","r_rra_30","r_rra_15","r_rra_10","r_rra_5","r_rra_2","r_rra_1",
    #         "t_rra_30","t_rra_15","t_rra_10","t_rra_5","t_rra_2","t_rra_1",]
    # total_ate = 0.0
    # total_rte = 0.0
    header_str = "Scene: "
    for key in keys:
        header_str += key + " "
    print(header_str)
    total_scenes = 0
    total_dict = {key: 0.0 for key in keys}
    for scene_folder in scene_folders:
        result_file = scene_folder / 'eval_result.json'
        gt_poses = scene_folder / 'gt.txt'
 
        # gt_poses = scene_folder / 'gt_ba.txt'
        aligned_poses = scene_folder / 'pred_ba.txt'
        ba_path = scene_folder/ "transforms.json"

            with open(ba_path, 'r') as f:
                transforms_json = json.load(f)
            transforms_json["frames"] = sorted(transforms_json["frames"], key=lambda x: x["file_path"])
            # print(
            #     [os.path.basename(fname["file_path"]) for fname in transforms_json["frames"]]
            # )
            c2ws_opengl = [frame["transform_matrix"] for frame in transforms_json['frames']]
            c2ws_opengl = [np.array(c2w).reshape(4, 4) for c2w in c2ws_opengl]
            c2ws_opencv = [convert_opengl_to_opencv(c2w,transform_type="cam2world") for c2w in c2ws_opengl]
            # c2ws_opencv = [convert_opengl_to_opencv(c2w,) for c2w in c2ws_opengl]
            # t1 = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
            # t2 = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
            # c2ws_opencv = [t1 @ c2w @ t2 for c2w in c2ws_opencv]
            c2ws_opencv = np.stack(c2ws_opencv, axis=0)

            import os
            scene_path = os.path.basename(str(scene_folder)).split("__")
            valid_mask,gt_poses_np = count_colmap_valid_mask(os.path.join("datasets/t2_colmap", scene_path[0], scene_path[1]), len(c2ws_opencv))
      
        # if not gt_poses.exists():
            if "training" not in str(scene_folder):
                # gt_poses = scene_folder / 'gt.txt'
                c2ws_opencv = [c2w for c2w, valid in zip(c2ws_opencv, valid_mask) if valid]
            c2ws_opencv = np.array(c2ws_opencv)
            # c2ws_opencv = np.linalg.inv(c2ws_opencv)
            # print(f"Number of valid poses: {valid_mask.sum()}")

            # c2ws_opencv = np.array(c2ws_opencv
            gt_poses_np = np.array(gt_poses_np)

            with open(aligned_poses, 'w') as f:
                for c2w in c2ws_opencv:
                    f.write(f"{c2w[0][0]} {c2w[0][1]} {c2w[0][2]} {c2w[0][3]} {c2w[1][0]} {c2w[1][1]} {c2w[1][2]} {c2w[1][3]} {c2w[2][0]} {c2w[2][1]} {c2w[2][2]} {c2w[2][3]}\n")
        if not gt_poses.exists():
            gt_poses = scene_folder / 'gt.txt'
            with open(gt_poses, 'w') as f:
                for c2w in gt_poses_np:
                    f.write(f"{c2w[0][0]} {c2w[0][1]} {c2w[0][2]} {c2w[0][3]} {c2w[1][0]} {c2w[1][1]} {c2w[1][2]} {c2w[1][3]} {c2w[2][0]} {c2w[2][1]} {c2w[2][2]} {c2w[2][3]}\n")
        else:
            aligned_poses = scene_folder / 'pred.txt'

        # confidence_depth_file
        confidence_depth = np.loadtxt(confidence_depth_file).reshape(-1, 18)[1:] if confidence_depth_file.exists() else None
        confidence_xyz = np.loadtxt(confidence_xyz_file).reshape(-1, 18)[1:] if confidence_xyz_file.exists() else None
        conf_filter=-1
        
        if confidence_xyz is not None:
            # conf_filter= confidence[:, 16] > 0.6
            max_conf_idx=17
            conf_filter= confidence_xyz[:, 16] > 2
            if (confidence_xyz[:, 1] > 0.6115).sum()<confidence_xyz.shape[0] * 0.99:
                    
                while max_conf_idx >= 0 and conf_filter.sum()<confidence_xyz.shape[0] * 0.288:
                    conf_filter |= confidence_xyz[:, max_conf_idx] > 0.6115
                    # conf_filter |= confidence_depth[:, max_conf_idx] > 0.8
                    max_conf_idx -= 1
            else:
                conf_filter = confidence_xyz[:, 5] > 0.6115
                
        

        # print(conf_filter)
        if not aligned_poses.exists():
            print(f"Results not found for {scene_folder.name}.")
            continue
        # aligned_poses = aligned_poses[:gt_poses.shape[0]]
        # try:
        eval_result = eval_pose_acc(gt_poses, aligned_poses, save_dir=scene_folder,conf_filter=conf_filter)
        # except Exception as e:
        #     print(f"Error evaluating poses for {scene_folder.name}: {e}")
            # continue
        if not result_file.exists():
            print(f"Results not found for {scene_folder.name}.")
            continue

        with open(result_file, 'r') as f:
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
    # print("Total:", )
    # print()   


