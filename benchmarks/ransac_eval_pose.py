import numpy as np
import sys
import torch
# sys.path.append('/home/users/junyuan.deng/jfs_public/Personal/cv/3/python/evo')
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from benchmarks.rotation import mat_to_quat

def save_kitti_poses(poses, save_path):
    with open(save_path, 'w') as f:
        for pose in poses:  # pose: 4x4 numpy array
            pose_line = pose[:3].reshape(-1)  # flatten first 3 rows
            f.write(' '.join(map(str, pose_line)) + '\n')

def align_gt_pred(gt_views, poses_c2w_estimated,correct_only_scale=False, correct_scale=True,n=-1):
    """
    对齐 GT 和预测的位姿，返回对齐后的预测位姿和尺度、旋转、平移参数。

    Args:
        poses_gt (np.ndarray): GT 位姿 (n, 4, 4).
        poses_pred (np.ndarray): 预测位姿 (n, 4, 4).

    Returns:
        Tuple:
    """
    # poses_c2w_gt = [view["camera_pose"][0] for view in gt_views]
    gt = PosePath3D(poses_se3=gt_views)
    pred = PosePath3D(poses_se3=poses_c2w_estimated)
    r_a, t_a, s = pred.align(gt, correct_scale=correct_scale,correct_only_scale=correct_only_scale,n=n)
    return pred.poses_se3, gt.poses_se3




import random
import math
import random
from collections import namedtuple
from scipy.spatial.transform import Rotation
import numpy as np
import logging
import cv2
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
def kabsch(pts1, pts2, estimate_scale=False):
    c_pts1 = pts1 - pts1.mean(axis=0)
    c_pts2 = pts2 - pts2.mean(axis=0)

    covariance = np.matmul(c_pts1.T, c_pts2) / c_pts1.shape[0]

    U, S, VT = np.linalg.svd(covariance)

    d = np.sign(np.linalg.det(np.matmul(VT.T, U.T)))
    correction = np.eye(3)
    correction[2, 2] = d

    if estimate_scale:
        pts_var = np.mean(np.linalg.norm(c_pts2, axis=1) ** 2)
        scale_factor = pts_var / np.trace(S * correction)
    else:
        scale_factor = 1.

    R = scale_factor * np.matmul(np.matmul(VT.T, correction), U.T)
    t = pts2.mean(axis=0) - np.matmul(R, pts1.mean(axis=0))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T, scale_factor
def get_inliers(h_T, poses_gt, poses_est, inlier_threshold_t, inlier_threshold_r):

    # h_T aligns ground truth poses with estimates poses
    poses_gt_transformed = h_T @ poses_gt

    # calculate differences in position and rotations
    translations_delta = poses_gt_transformed[:, :3, 3] - poses_est[:, :3, 3]
    rotations_delta = poses_gt_transformed[:, :3, :3] @ poses_est[:, :3, :3].transpose([0, 2, 1])

    # translation inliers
    inliers_t = np.linalg.norm(translations_delta, axis=1) < inlier_threshold_t
    # rotation inliers
    inliers_r = Rotation.from_matrix(rotations_delta).magnitude() < (inlier_threshold_r / 180 * math.pi)
    # intersection of both
    return np.logical_and(inliers_r, inliers_t)
def print_hyp(hypothesis, hyp_name):
    h_translation = np.linalg.norm(hypothesis['transformation'][:3, 3])
    h_angle = np.linalg.norm(Rotation.from_matrix(hypothesis['transformation'][:3, :3]).as_rotvec()) * 180 / math.pi
    print(f"{hyp_name}: score={hypothesis['score']}, translation={h_translation:.2f}m, "
                 f"rotation={h_angle:.1f}deg.")
def estimated_alignment(pose_est,pose_gt, inlier_threshold_t=0.05, inlier_threshold_r=5, ransac_iterations=1000, refinement_max_hyp=12, refinement_max_it=8, estimate_scale=False):

    n_pose = len(pose_est)
    ransac_hypotheses = []
    for i in range(ransac_iterations):
        min_sample_size=3
        samples = random.sample(range(n_pose), min_sample_size)
        h_pts1 = pose_gt[samples,:3,3]
        h_pts2 = pose_est[samples,:3,3]

        h_T, h_scale = kabsch(h_pts1, h_pts2, estimate_scale=estimate_scale)

        inliers = get_inliers(h_T, pose_gt, pose_est, inlier_threshold_t, inlier_threshold_r)

        if inliers[samples].sum() >= 3:
            # only keep hypotheses if minimal sample is all inliers
            ransac_hypotheses.append({
                "transformation": h_T,
                "inliers": inliers,
                "score": inliers.sum(),
                "scale": h_scale
            })
    if len(ransac_hypotheses) == 0:
        print(f"Did not fine a single valid RANSAC hypothesis, abort alignment estimation.")
        return None, 1

    # sort according to score
    ransac_hypotheses = sorted(ransac_hypotheses, key=lambda x: x['score'], reverse=True)

    # for hyp_idx, hyp in enumerate(ransac_hypotheses):
    #     print_hyp(hyp, f"Hypothesis {hyp_idx}")

    # create shortlist of best hypotheses for refinement
    # print(f"Starting refinement of {refinement_max_hyp} best hypotheses.")
    ransac_hypotheses = ransac_hypotheses[:refinement_max_hyp]

    # refine all hypotheses in the short list
    for ref_hyp in ransac_hypotheses:

        # print_hyp(ref_hyp, "Pre-Refinement")

        # refinement loop
        for ref_it in range(refinement_max_it):

            # re-solve alignment on all inliers
            h_pts1 = pose_gt[ref_hyp['inliers'], :3, 3]
            h_pts2 = pose_est[ref_hyp['inliers'], :3, 3]

            h_T, h_scale = kabsch(h_pts1, h_pts2, estimate_scale)

            # calculate new inliers
            inliers = get_inliers(h_T, pose_gt, pose_est, inlier_threshold_t, inlier_threshold_r)

            # check whether hypothesis score improved
            refined_score = inliers.sum()

            if refined_score > ref_hyp['score']:

                ref_hyp['transformation'] = h_T
                ref_hyp['inliers'] = inliers
                ref_hyp['score'] = refined_score
                ref_hyp['scale'] = h_scale

                # print_hyp(ref_hyp, f"Refinement interation {ref_it}")

            else:
                # print(f"Stopping refinement. Score did not improve: New score={refined_score}, "
                #              f"Old score={ref_hyp['score']}")
                break

    # re-sort refined hyotheses
    ransac_hypotheses = sorted(ransac_hypotheses, key=lambda x: x['score'], reverse=True)

    # for hyp_idx, hyp in enumerate(ransac_hypotheses):
        # print_hyp(hyp, f"Hypothesis {hyp_idx}")

    return ransac_hypotheses[0]['transformation'], ransac_hypotheses[0]['scale']

def eval_pose_ransac(gt, est, t_thres=0.05, r_thres=5, aligned=True, correct_scale=True, correct_scale_only=False, save_dir=None):

    if aligned:
        alignment_transformation, alignment_scale = estimated_alignment(est, gt, inlier_threshold_t=0.1, inlier_threshold_r=5, ransac_iterations=5000, refinement_max_hyp=12, refinement_max_it=16, estimate_scale=True)
        if alignment_transformation is None:
            _logger.info(f"Alignment requested but failed. Setting all pose errors to {math.inf}.")
    else:
        alignment_transformation = np.eye(4)
        alignment_scale = 1.
    # Evaluation Loop

    rErrs=[]
    tErrs=[]
    accuracy=0
    r_acc_5 = 0
    r_acc_2 = 0
    r_acc_1 = 0
    t_acc_15 = 0
    t_acc_10 = 0
    t_acc_5 = 0
    t_acc_2 = 0
    t_acc_1 = 0
    acc_10 = 0
    acc_5 = 0
    acc_2 = 0
    acc_1 = 0

    for pred_pose, gt_pose in zip(est, gt):

        if alignment_transformation is not None:
            # Apply alignment transformation to GT pose
            gt_pose = alignment_transformation @ gt_pose

            # Calculate translation error.
            t_err = float(np.linalg.norm(gt_pose[0:3, 3] - pred_pose[0:3, 3]))

            # Correct translation scale with the inverse alignment scale (since we align GT with estimates)
            t_err = t_err / alignment_scale

            # Rotation error.
            gt_R = gt_pose[0:3, 0:3]
            out_R = pred_pose[0:3, 0:3]

            r_err = np.matmul(out_R, np.transpose(gt_R))
            # Compute angle-axis representation.
            r_err = cv2.Rodrigues(r_err)[0]
            # Extract the angle.
            r_err = np.linalg.norm(r_err) * 180 / math.pi
        else:
            pose_gt = None
            t_err, r_err = math.inf, math.inf

        # _logger.info(f"Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.1f}cm")

        # Save the errors.
        rErrs.append(r_err)
        tErrs.append(t_err * 100) # in cm

        # Check various thresholds.
        if r_err < r_thres and t_err < t_thres:
            accuracy += 1
        if r_err < 5:
            r_acc_5 += 1
        if r_err < 2:
            r_acc_2 += 1
        if r_err < 1:
            r_acc_1 += 1
        if t_err < 0.15:
            t_acc_15 += 1
        if t_err < 0.10:
            t_acc_10 += 1
        if t_err < 0.05:
            t_acc_5 += 1
        if t_err < 0.02:
            t_acc_2 += 1
        if t_err < 0.01:
            t_acc_1 += 1    
        if r_err < 10 and t_err < 0.10:
            acc_10 += 1
        if r_err < 5 and t_err < 0.05:
            acc_5 += 1
        if r_err < 2 and t_err < 0.02:
            acc_2 += 1
        if r_err < 1 and t_err < 0.01:
            acc_1 += 1


    total_frames = len(rErrs)
    assert total_frames == len(est)

    # Compute median errors.
    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]

    # Compute final precision.
    accuracy = accuracy / total_frames * 100
    r_acc_5 = r_acc_5 / total_frames * 100
    r_acc_2 = r_acc_2 / total_frames * 100
    r_acc_1 = r_acc_1 / total_frames * 100
    t_acc_15 = t_acc_15 / total_frames * 100
    t_acc_10 = t_acc_10 / total_frames * 100
    t_acc_5 = t_acc_5 / total_frames * 100
    t_acc_2 = t_acc_2 / total_frames * 100
    t_acc_1 = t_acc_1 / total_frames * 100
    acc_10 = acc_10 / total_frames * 100
    acc_5 = acc_5 / total_frames * 100
    acc_2 = acc_2 / total_frames * 100
    acc_1 = acc_1 / total_frames * 100


    # _logger.info("===================================================")
    # _logger.info("Test complete.")

    # _logger.info(f'Accuracy: {accuracy:.1f}%')
    # _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")
    # print("===================================================")
    # print("Test complete.")

    with open(save_dir, "a") as f:

        f.write(f'Accuracy: {accuracy:.1f}%\n\n')
        f.write(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm\n")
        f.write(f'R acc 5: {r_acc_5:.1f}%\n')
        f.write(f'R acc 2: {r_acc_2:.1f}%\n')
        f.write(f'R acc 1: {r_acc_1:.1f}%\n')
        f.write(f'T acc 15: {t_acc_15:.1f}%\n')
        f.write(f'T acc 10: {t_acc_10:.1f}%\n')
        f.write(f'T acc 5: {t_acc_5:.1f}%\n')
        f.write(f'T acc 2: {t_acc_2:.1f}%\n')
        f.write(f'T acc 1: {t_acc_1:.1f}%\n')
        f.write(f'Acc 10: {acc_10:.1f}%\n')
        f.write(f'Acc 5: {acc_5:.1f}%\n')
        f.write(f'Acc 2: {acc_2:.1f}%\n')
        f.write(f'Acc 1: {acc_1:.1f}%\n')

def eval_rra_rta(gt, est,device="cpu"):
    gt_c2w = torch.from_numpy(gt).to(device)
    est = torch.from_numpy(gt).to(device)
    add_row = torch.tensor([0, 0, 0, 1], device=device).expand(gt_c2w.size(0), 1, 4)

    pred_se3 = closed_form_inverse_se3(torch.cat((est, add_row), dim=1))
    gt_se3 = torch.cat((gt_c2w, add_row), dim=1)

    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, len(pred_se3))
    return rel_rangle_deg, rel_tangle_deg

def build_pair_index(num_frames):
    return torch.meshgrid(torch.arange(num_frames), torch.arange(num_frames))

def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg
def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t
def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg

def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.
    This function assumes the input poses are world-to-camera (w2c) transformations.

    Args:
        pred_se3: Predicted SE(3) transformations (w2c), shape (N, 4, 4)
        gt_se3: Ground truth SE(3) transformations (w2c), shape (N, 4, 4)
        num_frames: Number of frames (N)

    Returns:
        Rotation and translation angle errors in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)
    pair_idx_i1 = pair_idx_i1.flatten()
    pair_idx_i2 = pair_idx_i2.flatten()
    # print(f"shape of gt_se3: {gt_se3.shape}, pred_se3: {pred_se3.shape}")
    # print(f"Number of pairs: {pair_idx_i1.shape}")
    # relative_pose_gt = gt_se3[pair_idx_i1].bmm(
    #     closed_form_inverse_se3(gt_se3[pair_idx_i2])
    # )
    # relative_pose_pred = pred_se3[pair_idx_i1].bmm(
    #     closed_form_inverse_se3(pred_se3[pair_idx_i2])
    # )
    relative_pose_gt = closed_form_inverse_se3(gt_se3[pair_idx_i2]).bmm(gt_se3[pair_idx_i1])
    relative_pose_pred = closed_form_inverse_se3(pred_se3[pair_idx_i2]).bmm(pred_se3[pair_idx_i1])
    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg

def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using NumPy.

    Args:
        r_error: numpy array representing R error values (Degree)
        t_error: numpy array representing T error values (Degree)
        max_threshold: Maximum threshold value for binning the histogram

    Returns:
        AUC value and the normalized histogram
    """
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram
import os
import argparse

def load_kitti_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))  # Add the last row for homogeneous coordinates
            poses.append(pose)
    return poses


def eval_pose_acc(gt_poses, pred_poses, save_dir=None,conf_filter=None):
    gt_poses = load_kitti_poses(gt_poses)
    aligned_poses = load_kitti_poses(pred_poses)

    assert len(gt_poses) ==  len(aligned_poses), f"Length of gt poses {len(gt_poses)} and aligned poses {len(aligned_poses)} do not match."

    
    gt_poses = np.stack(gt_poses, axis=0)
    aligned_poses = np.stack(aligned_poses, axis=0)
    
    gt_poses = torch.from_numpy(gt_poses).float()
    aligned_poses = torch.from_numpy(aligned_poses).float()
    # aligned_poses, gt_poses = align_gt_pred(gt_poses.numpy(), aligned_poses.numpy(),n=conf_filter)
    # gt_poses = np.stack(gt_poses, axis=0)
    # aligned_poses = np.stack(aligned_poses, axis=0)

    # gt_poses = torch.from_numpy(gt_poses).float()
    # aligned_poses = torch.from_numpy(aligned_poses).float()
    r_error, t_error = se3_to_relative_pose_error(aligned_poses, gt_poses, len(gt_poses))
    r_error = np.array(r_error)
    t_error = np.array(t_error)
    auc30 ,_ = calculate_auc_np(r_error, t_error, max_threshold=30)
    auc15 ,_ = calculate_auc_np(r_error, t_error, max_threshold=15)
    auc10 ,_ = calculate_auc_np(r_error, t_error, max_threshold=10)
    auc5 ,_ = calculate_auc_np(r_error, t_error, max_threshold=5)
    auc2 ,_ = calculate_auc_np(r_error, t_error, max_threshold=2)
    auc1 ,_ = calculate_auc_np(r_error, t_error, max_threshold=1)

    r_rra_30 = (r_error < 30).mean().item()
    r_rra_15 = (r_error < 15).mean().item()
    r_rra_10 = (r_error < 10).mean().item()
    r_rra_5 = (r_error < 5).mean().item()
    r_rra_2 = (r_error < 2).mean().item()
    r_rra_1 = (r_error < 1).mean().item()

    t_rta_30 = (t_error < 30).mean().item()
    t_rta_15 = (t_error < 15).mean().item()
    t_rta_10 = (t_error < 10).mean().item()
    t_rta_5 = (t_error < 5).mean().item() 
    t_rta_2 = (t_error < 2).mean().item()
    t_rta_1 = (t_error < 1).mean().item()

    gt_poses = gt_poses.numpy()
    aligned_poses = aligned_poses.numpy()
    gt_poses[...,:3,3] = gt_poses[...,:3,3] / np.linalg.norm(gt_poses[...,:3,3])
    aligned_poses, gt_poses = align_gt_pred(gt_poses, aligned_poses)
    ate_error=0
    # print(gt_poses[0].shape, aligned_poses[0].shape)
    aligned_poses = np.stack(aligned_poses, axis=0)
    gt_poses = np.stack(gt_poses, axis=0)
    ate_error = (((gt_poses[...,:3,3] - aligned_poses[...,:3,3])** 2).sum(axis=-1) ** 0.5).mean()



    ret_dict = {
        "ate":ate_error,"r_error": r_error.tolist(),
                "t_error": t_error.tolist(),
                "auc30": auc30,
                "auc15": auc15,
                "auc10": auc10,
                "auc5": auc5,
                "auc2": auc2,
                "auc1": auc1,
                "r_rra_30": r_rra_30,
                "r_rra_15": r_rra_15,          
                "r_rra_10": r_rra_10,
                "r_rra_5": r_rra_5,
                "r_rra_2": r_rra_2,
                "r_rra_1": r_rra_1,
                "t_rra_30": t_rta_30,
                "t_rra_15": t_rta_15,
                "t_rra_10": t_rta_10,       
                "t_rra_5": t_rta_5,
                "t_rra_2": t_rta_2,
                "t_rra_1": t_rta_1}
    
    import json
    with open(save_dir/"eval_result.json", "w") as f:
        json.dump(ret_dict, f, indent=4)
    return ret_dict
    # print(ret_dict)

if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_poses", type=str,  default="reconstructions/tnt_v6v3_30k_100_300/training__Barn/gt.txt", help="Path to the ground truth poses file.")
    parser.add_argument("--aligned_poses", type=str, default="reconstructions/tnt_v6v3_30k_100_300/training__Barn/pred.txt", help="Path to the aligned predicted poses file.")
    parser.add_argument("--norm_gt_pose", action="store_true", help="norm translation of gt pose to tr(XX^\{T\})=1") # type: ignore
    parser.add_argument("--save_dir", type=str, default="reconstructions/tnt_v6v3_30k_100_300/training__Barn/", help="Directory to save the evaluation results.")
    args = parser.parse_args()

    gt_poses = load_kitti_poses(args.gt_poses)
    aligned_poses = load_kitti_poses(args.aligned_poses)

    assert len(gt_poses) ==  len(aligned_poses), f"Length of gt poses {len(gt_poses)} and aligned poses {len(aligned_poses)} do not match."

    gt_poses = np.stack(gt_poses, axis=0)
    aligned_poses = np.stack(aligned_poses, axis=0)
    gt_poses = torch.from_numpy(gt_poses).float()
    aligned_poses = torch.from_numpy(aligned_poses).float()
    r_error, t_error = se3_to_relative_pose_error(aligned_poses, gt_poses, len(gt_poses))
    r_error = np.array(r_error)
    t_error = np.array(t_error)
    auc30 ,_ = calculate_auc_np(r_error, t_error, max_threshold=30)
    auc15 ,_ = calculate_auc_np(r_error, t_error, max_threshold=15)
    auc10 ,_ = calculate_auc_np(r_error, t_error, max_threshold=10)
    auc5 ,_ = calculate_auc_np(r_error, t_error, max_threshold=5)
    auc2 ,_ = calculate_auc_np(r_error, t_error, max_threshold=2)
    auc1 ,_ = calculate_auc_np(r_error, t_error, max_threshold=1)

    r_rra_30 = (r_error < 30).mean().item()
    r_rra_15 = (r_error < 15).mean().item()
    r_rra_10 = (r_error < 10).mean().item()
    r_rra_5 = (r_error < 5).mean().item()
    r_rra_2 = (r_error < 2).mean().item()
    r_rra_1 = (r_error < 1).mean().item()

    t_rta_30 = (t_error < 30).mean().item()
    t_rta_15 = (t_error < 15).mean().item()
    t_rta_10 = (t_error < 10).mean().item()
    t_rta_5 = (t_error < 5).mean().item() 
    t_rta_2 = (t_error < 2).mean().item()
    t_rta_1 = (t_error < 1).mean().item()

    ret_dict = {"r_error": r_error.tolist(),
                "t_error": t_error.tolist(),
                "auc30": auc30,
                "auc15": auc15,
                "auc10": auc10,
                "auc5": auc5,
                "auc2": auc2,
                "auc1": auc1,
                "r_rra_30": r_rra_30,
                "r_rra_15": r_rra_15,          
                "r_rra_10": r_rra_10,
                "r_rra_5": r_rra_5,
                "r_rra_2": r_rra_2,
                "r_rra_1": r_rra_1,
                "t_rra_30": t_rta_30,
                "t_rra_15": t_rta_15,
                "t_rra_10": t_rta_10,       
                "t_rra_5": t_rta_5,
                "t_rra_2": t_rta_2,
                "t_rra_1": t_rta_1}
    
    # import json
    # with open(os.path.join(args.save_dir, "eval_result.json"), "w") as f:
    #     json.dump(ret_dict, f, indent=4)
    # print(ret_dict)


    # aligned_poses, gt_poses = align_gt_pred(gt_poses, aligned_poses)

    # save_kitti_poses(
    #     gt_poses,
    #     os.path.join(save_dir, "gt.txt"),
    # )
    # save_kitti_poses(
    #     aligned_poses,
    #     os.path.join(save_dir, "pred.txt"),
    # )
    # result_path = os.path.join(args.save_dir, "result.txt")

    # eval_pose_ransac(
    #     np.stack(gt_poses),
    #     np.stack(aligned_poses),
    #     save_dir=args.save_dir
    # )
    # r_err_5 = 0
    # r_err_2 = 0
    # pct15_15 = 0
    # pct10_10 = 0
    # pct10_5 = 0
    # pct5 = 0
    # pct2 = 0
    # pct1 = 0
    # total_frames = len(gt_poses)
    # total = 0
    # for i, (a_p, gt_p) in enumerate(zip(torch.from_numpy(np.array(aligned_poses)), torch.from_numpy(np.array(gt_poses)))):
    #     t_err = float(torch.norm(a_p[0:3, 3] - gt_p[0:3, 3]))

    #     gt_R = gt_p[0:3, 0:3].numpy()
    #     out_R = a_p[0:3, 0:3].numpy()

    #     #R_fix = np.diag([1, -1, 1])
    #     # out_R = R_fix @ out_R

    #     r_err = np.matmul(out_R, np.transpose(gt_R))
    #     # Compute angle-axis representation.
    #     r_err = cv2.Rodrigues(r_err)[0]
    #     # Extract the angle.
    #     r_err = np.linalg.norm(r_err) * 180 / math.pi

    #     if r_err < 5:
    #         r_err_5 += 1
    #     if r_err < 2:
    #         r_err_2 += 1
    #     if r_err < 15 and t_err < 0.15:  # 10cm/10deg
    #         pct15_15 += 1  
    #     if r_err < 10 and t_err < 0.1:  # 10cm/10deg
    #         pct10_10 += 1
    #     if r_err < 5 and t_err < 0.1:  # 10cm/5deg
    #         pct10_5 += 1
    #     if r_err < 5 and t_err < 0.05:  # 5cm/5deg
    #         pct5 += 1
    #     if r_err < 2 and t_err < 0.02:  # 2cm/2deg
    #         pct2 += 1
    #     if r_err < 1 and t_err < 0.01:  # 1cm/1deg
    #         pct1 += 1
    # # result_path = os.path.join(args.save_dir, "result.txt")

    # with open(args.save_dir, "a") as f:
    # f.write("\n\n")
    # # f.write(f"val_reloc/reloc_time {reloc_time:.4f}\n")
    # # f.write(f"fps {total_frames / reloc_time:.4f}\n")
    # f.write(f"val_reloc/r_err_5 {r_err_5 / total_frames * 100:.4f}\n")
    # f.write(f"val_reloc/r_err_2 {r_err_2 / total_frames * 100:.4f}\n")
    # f.write(f"val_reloc/pct15_15 {pct15_15 / total_frames * 100:.4f}\n")
    # f.write(f"val_reloc/pct10_10 {pct10_10 / total_frames * 100:.4f}\n")
    # f.write(f"val_reloc/pct10_5 {pct10_5 / total_frames * 100:.4f}\n")
    # f.write(f"val_reloc/pct5 {pct5 / total_frames * 100:.4f}\n")
    # f.write(f"val_reloc/pct2 {pct2 / total_frames * 100:.4f}\n")
    # f.write(f"val_reloc/pct1 {pct1 / total_frames * 100:.4f}\n")