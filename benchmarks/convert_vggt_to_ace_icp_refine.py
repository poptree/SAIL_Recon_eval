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
import open3d as o3d
from benchmarks.preprocess_data import Resolution, Frame, glob_for_frames, get_resolution_from_frames

def incremental_alignment_icp(source_points, target_points, max_iterations=1000, tolerance=1e-6):
    pass



def convert_tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError(f"Unsupported type: {type(tensor)}. Expected torch.Tensor or np.ndarray.")

def filter_depth_by_confidence(depth, confidence, threshold=0.5):
    if confidence is None:
        return depth
    depth[confidence<threshold] = 0.0
    return depth

def convert_tensor_rgbd_to_open3d_point_cloud(rgb, depth, intrinsic, extrinsic, confidence=None,depth_threshold=2.0):
    depth = filter_depth_by_confidence(depth, confidence, threshold=depth_threshold)
    rgb = convert_tensor_to_numpy(rgb)
    depth = convert_tensor_to_numpy(depth)
    intrinsic = convert_tensor_to_numpy(intrinsic)
    extrinsic = convert_tensor_to_numpy(extrinsic)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth),
        depth_scale=1.0,
        depth_trunc=10,
        convert_rgb_to_intensity=False
    )
    
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsic.shape[1],
        height=intrinsic.shape[0],
        fx=intrinsic[0, 0],
        fy=intrinsic[1, 1],
        cx=intrinsic[0, 2],
        cy=intrinsic[1, 2]
    )
    pcd_o3d_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d)
    pcd_o3d_world = pcd_o3d_cam.transform(np.linalg.inv(extrinsic))

    return pcd_o3d_world

def alignment_icp(source_points, target_points, init_transform=None, max_iterations=1000, tolerance=1e-6, *args, **kwargs):
    source_points = source_points.estimate_normals()
    target_points = target_points.estimate_normals()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=100,  # 最大迭代次数
        relative_fitness=1e-6,  # 相对fitness变化
        relative_rmse=1e-6)     # 相对RMSE变化
    
    result = o3d.pipelines.registration.registration_colored_icp(
        source_points, target_points,
        0.01,
        init=init_transform if init_transform else np.eye(4),  # 初始变换矩阵(单位矩阵)
        criteria=criteria)

    return result


class Open3DRGBDICP:
    def __init__(self, max_iterations=1000, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def align(self, source_points, target_points, *args, **kwargs):
        return alignment_icp(source_points, target_points, self.max_iterations, self.tolerance, *args, **kwargs)
    
    def run_alignment(self, tensor_npzs, incremental_idx=[]):
    
        incremental_idx = incremental_idx if incremental_idx is not None else range(len(tensor_npzs))


        incremental_deform_poses = []
        incremental_pcd = None
        for cur_i in incremental_idx:
            frame_meta = torch.load(tensor_npzs[cur_i]['meta'])
            rgb_tensor = frame_meta['rgb']
            depth_tensor = frame_meta['depth'][...,0]
            depth_confidence = frame_meta['depth'][...,1]
            intrinsic_tensor = frame_meta['intrinsic']
            extrinsic_tensor = frame_meta['extrinsic']
            

            if incremental_pcd is None:

                incremental_pcd = convert_tensor_rgbd_to_open3d_point_cloud(
                    rgb_tensor, depth_tensor, intrinsic_tensor, extrinsic_tensor, confidence=depth_confidence,
                )
            

            frame_pcd = convert_tensor_rgbd_to_open3d_point_cloud(
                rgb_tensor, depth_tensor, intrinsic_tensor, extrinsic_tensor, confidence=depth_confidence,
            )
            pose_transform = alignment_icp(frame_pcd, incremental_pcd, max_iterations=self.max_iterations, tolerance=self.tolerance, init_transform=np.eye(4) if len(incremental_deform_poses)==0 else incremental_deform_poses[-1])
            incremental_pcd = incremental_pcd + frame_pcd.transform(pose_transform.transformation)
            incremental_deform_poses.append(pose_transform.transformation)
        len_frames =  50
        all_frames_deform_poses = []
        for cur_i in range(len_frames):
            frame_meta = torch.load(tensor_npzs[cur_i]['meta'])
            rgb_tensor = frame_meta['rgb']
            depth_tensor = frame_meta['depth'][...,0]
            depth_confidence = frame_meta['depth'][...,1]
            intrinsic_tensor = frame_meta['intrinsic']
            extrinsic_tensor = frame_meta['extrinsic']

            frame_pcd = convert_tensor_rgbd_to_open3d_point_cloud(
                rgb_tensor, depth_tensor, intrinsic_tensor, extrinsic_tensor, confidence=depth_confidence,
            )

            pose_transform = alignment_icp(frame_pcd, incremental_pcd, max_iterations=self.max_iterations, tolerance=self.tolerance, init_transform=np.eye(4) if len(all_frames_deform_poses)==0 else all_frames_deform_poses[-1])

            all_frames_deform_poses.append(pose_transform.transformation)

        return all_frames_deform_poses



def matrix_to_quaternion(matrix):
    R_matrix = matrix[:3, :3]
    r = R.from_matrix(R_matrix)
    # q = r.as_quat(scalar_first=True)  # Convert to quaternion (w,x,y,z)
    q = r.as_quat()  # Convert to quaternion (x,y,z,w)
    q = np.concatenate([q[1:], q[:1]])  # Reorder to (w,x,y,z)
    t = matrix[:3, 3]
    return q,t

def convert_vggt_poses_to_ace_poses(data_pattern:str, pose_file: str, intrinsics_file: str, intrinsics_type:str, confidence_depth_file: Optional[str], confidence_point_file: Optional[str], output_file:str, split_json: Optional[str], split_set:str,) -> None:
    
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
        confidence_depth = np.loadtxt(confidence_depth_file).reshape(-1, 18)
        confidence_depth = confidence_depth[1:]  # Assuming the first column is the confidence value
    if confidence_point_file is not None:
        print(f"Confidence point file: {confidence_point_file}")
        confidence_point = np.loadtxt(confidence_point_file).reshape(-1, 18)
        confidence_point = confidence_point[1:]  # Assuming the first column is the confidence value
    
    # print(dataset_frames)

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
        split_pick = []
        for i, frame in enumerate(dataset_frames):
            if str(frame.rgb_path) in split_frames:
                split_pick.append(True)
            else:
                split_pick.append(False)
        dataset_frames=[f for f in dataset_frames if str(f.rgb_path) in split_frames]
        if confidence_depth_file is not None:
            confidence_depth = confidence_depth[split_pick]
        if confidence_point_file is not None:
            confidence_point = confidence_point[split_pick]
    

    confidences = []
    if confidence_depth_file is  None and confidence_point_file is  None:
        confidences = [1001 for _ in range(len(dataset_frames))]     
    elif confidence_depth_file is not None :
        confidences = [1001 for _ in range(len(dataset_frames))] 
    elif confidence_point_file is not None:
        median_1_25_95 = np.sort(confidence_point[:,1])[int(len(confidence_point[:,1])*0.8)]
        median_1_50_95 = np.sort(confidence_point[:,2])[int(len(confidence_point[:,2])*0.8)]
        def confidence_filter(confidence, median_1_25_95, median_1_50_95):
            if confidence[0] < 0.2:
                return 500
            if confidence[1] < 0.1:
                return 500
            if confidence[2] < 0.01:
                return 500
            if confidence[1] < median_1_25_95 or confidence[2] < median_1_50_95:
                return 500
            return 1000
        confidences = [confidence_filter(c, median_1_25_95, median_1_50_95) for c in confidence_point]
        
        
    # for frame in dataset_frames:
    #     confidences.append(1001)
    fx = np.median(K33[...,0,0])
    # print(confidences)
    with open(output_file, 'w') as f:
        for frame,q,t,k,c in zip(dataset_frames, qs,ts, K33, confidences):
            # print(f"Processing frame: {frame.rgb_path}")
            f.write(f"{frame.rgb_path} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {fx} {c}\n")
    with open(os.path.join(os.path.split(output_file)[0],"focal_length.txt"), 'w') as f:
        f.write(f"{fx}\n")

    # print(f"Confidence point file: {confidence_point_file}")
    # print (f"Confidence point file: {confidence_point_file}, confidence length: {len(confidences)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="convert vggt poses to ace format")
    parser.add_argument("--pred_poses", type=str, required=True, help="the predicted poses file in vggt format")
    parser.add_argument("--intrinsics", type=str, required=True, help="the intrinsics file in vggt format")
    parser.add_argument("--data_pattern", type=str, required=True, help="the glob pattern for the input data")
    parser.add_argument("--output_file", type=str, required=True, help="the output file in ace format")


    parser.add_argument("--confidence_depth_file", type=str, required=False,
                        help="the confidence depth file in vggt format")
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
                                     confidence_point_file=args.confidence_depth_file,
                                     output_file=args.output_file,
                                     split_json=args.split_json,
                                     split_set=args.split_set,
                                     )
    print(args.confidence_depth_file)
    # print(args.confidence_point_file)