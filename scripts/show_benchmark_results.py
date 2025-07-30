import argparse
import os
import json
from pathlib import Path

# Parse command line argument for folder name
parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path,
                    help="Folder containing Nerfstudio bechmark results with subfolders for each scene.")
parser.add_argument("--method", type=str, default='nerfacto', choices=['nerfacto', 'splatfacto'],
                    help="Method of Nerfstudio used for benchmarking.")
args = parser.parse_args()

# metrics to report from the eval.json file
keys = ['psnr', 'ssim', 'lpips']

# get list of scenes as sub folders of benchmarking folder
scene_folders = sorted([f for f in args.folder.iterdir() if f.is_dir()])

# Print header
header_str = "Scene: "
for key in keys:
    header_str += key + " "
print(header_str)

total_psnr = 0.0
total_ssim = 0.0
total_lpips = 0.0
total_scenes = 0
# Loop through scenes of dataset
for scene_folder in scene_folders:
    # Specify result file
    result_file = scene_folder / f'nerf_data/nerf_for_eval/{args.method}/run/eval.json'

    # Assemble an output string with the benchmarking results
    out_str = scene_folder.name + ": "

    # Check whether result file exists
    if not os.path.exists(result_file):
        out_str += "Results not found."
    else:
        # Load results
        with open(os.path.join(result_file), 'r') as f:
            data = json.load(f)

            # Print all requested values
            for key in keys:
                if key in data['results']:
                    out_str += str(data['results'][key]) + " "
                    if key == 'psnr':
                        total_psnr += data['results'][key]
                    elif key == 'ssim':
                        total_ssim += data['results'][key]
                    elif key == 'lpips':
                        total_lpips += data['results'][key]
                else:
                    out_str += "Invalid Key "
            total_scenes += 1

    print(out_str)
print("Total: ", total_psnr, total_ssim, total_lpips)
print("Average : ", total_psnr / total_scenes if total_scenes > 0 else 0, total_ssim / total_scenes if total_scenes > 0 else 0, total_lpips / total_scenes if total_scenes > 0 else 0)