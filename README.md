# SAIL-Recon Eval

Please follow the instruction below to setup the enviorment.

```bash
# Runing PSNR benchmark
bash scripts/scripts/eval_sailrecon_7scenes.sh
bash scripts/scripts/eval_sailrecon_mip360.sh
bash scripts/scripts/eval_sailrecon_tnt_advanced_videos.sh
bash scripts/scripts/eval_sailrecon_tnt_advanced.sh
bash scripts/scripts/eval_sailrecon_tnt_intermediate_videos.sh
bash scripts/scripts/eval_sailrecon_tnt_intermediate.sh
bash scripts/scripts/eval_sailrecon_tnt_training_videos.sh
bash scripts/scripts/eval_sailrecon_tnt_training.sh

# show result of PSNR benchmark
python scripts/show_benchmark_results.py benchmark_output/sail-recon/tnt_training_videos

# show result of pose benchmark
python benchmarks/eval_sailrecon_pose.py --dir reconstructions/sail-recon
```


# ACE0 (ACE Zero)

This repository contains the code associated to the ACE0 paper:
> **Scene Coordinate Reconstruction: 
Posing of Image Collections via Incremental Learning of a Relocalizer**
> 
> [Eric Brachmann](https://ebrach.github.io/), 
> [Jamie Wynn](https://scholar.google.com/citations?user=ASP-uu4AAAAJ&hl=en), 
> [Shuai Chen](https://chenusc11.github.io/), 
> [Tommaso Cavallari](https://scholar.google.it/citations?user=r7osSm0AAAAJ&hl=en), 
> [Áron Monszpart](https://amonszpart.github.io/), 
> [Daniyar Turmukhambetov](https://dantkz.github.io/about/), and 
> [Victor Adrian Prisacariu](https://www.robots.ox.ac.uk/~victor/)
> 
> ECCV 2024, Oral

For further information please visit:

- [Project page (with a method overview and videos)](https://nianticlabs.github.io/acezero)
- [Arxiv](https://arxiv.org/abs/2404.14351)

Table of contents:

- [Installation](#installation)
- [Usage](#usage)
    - [Basic Usage](#basic-usage)
    - [Visualization Capabilities](#visualization-capabilities)
    - [Advanced Use Cases](#advanced-use-cases)
      - [Refine Existing Poses](#refine-existing-poses)
      - [Start From a Partial Reconstruction](#start-from-a-partial-reconstruction)
      - [Self-Supervised Relocalization](#self-supervised-relocalization)
      - [Train NeRF models or Gaussian splats](#train-nerf-models-or-gaussian-splats)
    - [Utility Scripts](#utility-scripts)
- [Benchmark](#benchmark)
  - [Nerfacto](#nerfacto)
  - [Splatfacto](#splatfacto)
- [Paper Experiments](#paper-experiments)
  - [7-Scenes](#7-scenes)
  - [Mip-NeRF 360](#mip-nerf-360)
  - [Tanks and Temples](#tanks-and-temples)
- [Frequently Asked Questions](#frequently-asked-questions)
- [References](#publications)

## Installation

This code uses PyTorch and has been tested on Ubuntu 20.04 with a V100 Nvidia GPU, although it should reasonably run 
with other Linux distributions and GPUs as well. Note our [FAQ](#frequently-asked-questions) if you want to run ACE0 on GPUs with less memory. 

We provide a pre-configured [`conda`](https://docs.conda.io/en/latest/) environment containing all required dependencies
necessary to run our code.
You can re-create and activate the environment with:

```shell
conda env create -f environment.yml
conda activate ace0
```

**All the following commands in this file need to run from the repository root and in the `ace0` environment.**

ACE0 represents a scene using an [ACE](https://nianticlabs.github.io/ace/) scene coordinate regression model.
In order to register cameras to the scene, it relies on the RANSAC implementation of the DSAC* paper (Brachmann and
Rother, TPAMI 2021), which is written in C++.
As such, you need to build and install the C++/Python bindings of those functions.
You can do this with:

```shell
cd dsacstar
python setup.py install
cd ..
```

Having done the steps above, you are ready to experiment with ACE0!

**Important note:** the first time you run ACE0, the script may ask you to confirm that you are happy to download the ZoeDepth depth estimation code and its pretrained weights from GitHub.
See [this link](https://github.com/isl-org/ZoeDepth) for its license and details.
ACE0 uses that model to estimate the depth for the seed images.
It can be replaced, please see the [FAQ](#frequently-asked-questions) section below for details.

## Docker

If you would prefer to run Ace0 in a docker container, you can start it with:

```shell  
docker-compose up -d 
```

You can then shell into the container with the following command: 

```shell  
docker exec -it acezero /bin/bash
```

From there you can follow the Gaussian Splatting tutorial described at the bottom of the README [here.](#frequently-asked-questions) Make sure to add your images to the volume defined in ```docker-compose.yml```

## Usage

We explain how to run ACE0 to reconstruct images from scratch, with and without knowledge about the image intrinsics.
We also explain how to use ACE0 to refine existing poses, or to initialise reconstruction with a subset of poses.
Furthermore, we cover the visualization capabilities of ACE0, including export of the reconstruction as a video and as 
3D models.

### Basic Usage

In the minimal case, you can run ACE0 on a set of images as defined by a glob pattern. 

```shell
# running on a set of images with default parameters
python ace_zero.py "/path/to/some/images/*.jpg" result_folder
```

Note the quotes around the glob pattern to ensure it is passed to the ACE0 script rather than being expanded by the shell.

If you want to run ACE0 on a video, you can extract frames from the video and run ACE0 on the extracted frames, see our [Utility Scripts](#utility-scripts).

The ACE0 script will call ACE training (`train_ace.py`) and camera registration (`register_mapping.py`) in a loop until 
all images have been registered to the scene representation, or there is no change between iterations.

The result of an ACE0 reconstruction is the `poses_final.txt` in the result folder. 
These files contain the estimated image poses in the following format:
```
filename qw qx qy qz x y z focal_length confidence
``` 
`filename` is the image file relative to the repository root.
`qw qx qy qz ` is the camera rotation as a quaternion, and `x y z` is the camera translation.
Camera poses are world-to-camera transformations, using the OpenCV camera convention.
`focal_length` is the focal length estimated by ACE0 or set externally (see below).
`confidence` is the reliability of an estimate. 
If the confidence is less than 1000, it should be treated as unreliable and possibly ignored.

The pose files can be used e.g. to train a Nerfacto or Splatfacto model, using our benchmarking scripts, see [Benchmarking](#benchmark).
Our benchmarking scripts also allow you to only convert our pose files to the format required by Nerfstudio, without running the benchmark itself.

<details>
<summary>Other content of the result folder explained.</summary>

The result folder will contain files such as the following:

- `iterationX.pt`: The ACE scene model (the MLP network) at iteration X. Output of `train_ace.py` in iteration X.
- `iterationX.txt`: Training statistics of the ACE model at iteration X, e.g. loss values, pose statistics, etc. See `ace_trainer.py`. Output of `train_ace.py` in iteration X.
- `poses_iterationX_preliminary.txt`: Poses of cameras after the mapping iteration but before relocalization. Contains poses refined by the MLP, rather than poses re-estimated by RANSAC. Output of `train_ace.py` in iteration X. 
- `poses_iterationX.txt`: Final poses of iteration X, after relocalization, i.e. re-estimated by RANSAC. Output of `register_mapping.py` in iteration X.
- `poses_final.txt`: The final poses of the images in the scene. Corresponds to the poses of the last relocalisation iteration, i.e. the output of the last `register_mapping.py` call.
- `pc_final.ply`: An ACE0 point cloud of the scene, for visualisation or initialisation of Gaussian splats. This output is optional and triggered using the `--export_point_cloud True` option of `ace_zero.py`.
</details>

#### Setting Calibration Parameters

Using default parameters, ACE0 will estimate the focal length of the images, starting from a heuristic value (70% of the image diagonal.)
If you have a better estimate of the focal length, you can provide it as an initialisation parameter.

```shell
# running ACE0 with an initial guess for the focal length
python ace_zero.py "/path/to/some/images/*.jpg" result_folder --use_external_focal_length <focal_length>
```

Using the call above, ACE0 will refine the focal length throughout the reconstruction process.
If you are confident that your focal length value is correct, you can disable focal length refinement.

```shell
# running ACE0 with a fixed focal length
python ace_zero.py "/path/to/some/images/*.jpg" result_folder --use_external_focal_length <focal_length> --refine_calibration False
```

**Note:** The current implementation of ACE0 supports only a single focal length value shared by all images. 
ACE0 currently also does assume that the principal point is at the image center, and pixels are square and unskewed.
Changing these assumptions should be possible, but requires some implementation effort.

### Visualization Capabilities

ACE0 can visualize the reconstruction process as a video. 

```shell
# running ACE0 with visualisation enabled
python ace_zero.py "/path/to/some/images/*.jpg" result_folder --render_visualization True
```

With visualisation enabled, ACE0 will render individual frames in a subfolder `renderings` and call `ffmpeg` at the end.
The visualisation will be saved as a video in the results folder, named `reconstruction.mp4`.

<details>
<summary>Other content of the renderings folder explained.</summary>

* `frame_N.png`: The Nth frame of the video.
* `iterationX_mapping.pkl`: The visualisation buffer of the mapping call in iteration X. It stores the 3D point cloud of the scene, the last rendering camera for a smooth transition, and the last frame index.
* `iterationX_register.pkl`: The visualisation buffer of the relocalization call in iteration X.
</details>

**Note that this will slow down the reconstruction considerably.**
Alternatively, you can run without visualisation enabled and export the final reconstruction as a 3D model, see [Utility Scripts](#utility-scripts).

### Advanced Use Cases

You can combine the ACE0 meta script with custom calls to `train_ace.py` and `register_mapping.py` to cater to more advanced use cases.

* `train_ace.py`: Trains an ACE model on a set of images with corresponding poses.
* `register_mapping.py`: Estimates poses of images in a scene given an ACE model.
* `ace_zero.py`: Can start from an existing ACE model.

You are free to switch image sets between the calls to these functions.
We provide some examples of advanced use cases that also cover some of the experiments in our paper.

#### Refine Existing Poses

If you have an initial guess of all image poses, you can use ACE to refine them quickly.
We combine a single ACE mapping call with pose refinement enabled, and a single relocalization call.

```shell
# running ACE mapping with pose refinement enabled
python train_ace.py "/path/to/some/images/*.jpg" result_folder/ace_network.pt --pose_files "/path/to/some/images/*.txt" --pose_refinement mlp --pose_refinement_wait 5000 --use_external_focal_length <focal_length> --refine_calibration False

# re-estimate poses of all images
python register_mapping.py "/path/to/some/images/*.jpg" result_folder/ace_network.pt --use_external_focal_length <focal_length> --session ace_network
```

In this example, ACE takes the existing poses in the [7-Scenes](#7-scenes) format as input: one text file per image with the camera-to-world pose stored as a 4x4 matrix.
The option `--pose_refinement mlp` enables pose refinement using a refinement network.
The option `--pose_refinement_wait 5000` freezes poses for the first 5000 iterations which increases the stability if you are mapping from scratch with pose refinement.

After calling `register_mapping.py`, the result folder will contain the refined poses in `poses_ace_network.txt`.

Note that the example above assumes a known, fixed focal length. If you let ACE refine the calibration, you need to pass the refined focal length of `train_ace.py` to `register_mapping.py`.
Please see `scripts/reconstruct_7scenes_warmstart.sh` for a complete example where we refine KinectFusion poses with ACE.

#### Start From a Partial Reconstruction

If you have pose estimates for subsets of images, you can use ACE0 to complete the reconstruction.
First, you call ACE mapping on the subset of images with poses which results in an ACE scene model.
You pass this model to ACE0, which will then register the remaining images to the scene.

```shell
# running ACE mapping on a subset of images wit poses
python train_ace.py "/images/with/poses/*.jpg" result_folder/iteration0_seed0.pt --pose_files "/poses/of/images/*.txt" --use_external_focal_length <focal_length> --refine_calibration False

# running ACE0 with the ACE model as a seed, and the complete set of images
python ace_zero.py "/all/images/*.jpg" result_folder --seed_network result_folder/iteration0_seed0.pt --use_external_focal_length ${focal_length} --refine_calibration False
```

ACE0 will store the final poses in `poses_final.txt` in the result folder, containing poses of all images.
Note that the example above assumes a known, fixed focal length.
You can also let ACE or ACE0 estimate or refine the focal length, but you need to take care of passing the correct focal length between the calls.

Please see `scripts/reconstruct_t2_training_videos_warmstart.sh` for a complete example where we reconstruct the Tanks and Temples training scenes starting from a partial reconstruction by COLMAP. More information about this example in [Tanks and Temples](#tanks-and-temples).

#### Self-Supervised Relocalization

You can use ACE0 to map a set of images, and call `register_mapping.py` on a different set of images to relocalize them.
Here, ACE0 would run on the set of mapping images, while `register_mapping.py` would run on the set of query images.

```shell
# running ACE0 on the mapping images
python ace_zero.py "/path/to/mapping/images/*.jpg" result_folder --use_external_focal_length <focal_length> --refine_calibration False

# running relocalization on the query images
python register_mapping.py "/path/to/query/images/*.jpg" result_folder/iterationX.pt --use_external_focal_length <focal_length> --session query
```

You need to point `register_mapping.py` to the ACE model from the last mapping iteration (e.g. `iterationX.pt`). 
The relocalization results will be stored in `poses_query.txt`.
Note that ACE0 reconstructions are only approximately metric. 
If you compare the query poses to ground truth, you need to fit a similarity transform first.
We provide a script for doing that.

```shell
python eval_poses.py result_folder/poses_query.txt "/path/to/ground/truth/poses/*.txt"
```

More information about the evaluation script can be found under [Utility Scripts](#utility-scripts).

#### Train NeRF models or Gaussian splats

See [Benchmarking](#benchmark) for instructions on how to use Nerfstudio on top of ACE0.

### Utility Scripts

#### Video to Dataset

We provide a script for extracting frames from MP4 videos via ffmpeg.

```shell
python datasets/video_to_dataset.py datasets
```

The script looks for all MP4 files in the target folder (here `datasets`) and extracts frames into a subfolder `datasets/video_<mp4_file_name>` for each video.

#### Export 3D Scene as Point Cloud

We provide a script for exporting ACE point clouds from a network and a pose file.

```shell
python export_point_cloud.py point_cloud_out.txt --network /path/to/ace_network.pt --pose_file /path/to/poses_final.txt
````

The script can either write out TXT of PLY files, decided by the file extension of the output file you specify. 
If the output file has a .txt extension, the script will write the point cloud into a text file in the format `x y z r g b` per line for each point.
If the output file has a .ply extension, the script will write the point cloud into a binary PLY file.
Both formats can be imported into most 3D software, e.g. Meshlab, CloudCompare, etc.
The PLY format is understood by Nerfstudio for initialisation of Gaussian splats.
Note, you can also point the script to an existing visualization buffer, `result_folder/renderings/iterationX_mapping.pkl`, which already contains the point cloud so it does not have to be re-generated.

Point clouds can be exported either using OpenGL or OpenCV coordinate conventions. Nerfstudio expects OpenCV coordinates.
The script can extract sparse or dense point clouds. The sparse point clouds have more filters applied and look cleaner.
The dense point clouds tend to work better for Gaussian splatting if you have a lot of images (2000+) as they cover more of the background. 

#### Export Cameras as Mesh

We provide a script for exporting an ACE pose file to PLY showing the cameras.

```shell
python export_cameras.py /path/to/ace/pose_file.txt /path/to/output.ply
```

The script will color-code the cameras by their confidence value. 
The PLY format can be imported into most 3D software, e.g. Meshlab, CloudCompare, etc.

#### Evaluate Poses Against (Pseudo) Ground Truth

We provide a script that measures the pose error of a set of estimated poses against a set of ground truth poses.

```shell
python eval_poses.py /path/to/ace/pose_file.txt "/path/to/ground/truth/poses/*.txt"
```

The ground truth poses are given as a glob pattern, where each file contains the pose of a single image as a 4x4 camera-to-world transformation (e.g. as provided by the 7-Scenes dataset).
Correspondence between ACE estimates and ground truth files will be established via alphabetical order of the image filenames.

By default, the script will calculate the percentage of poses below 5cm and 5 degrees error, as well as median rotation and translation errors.
Since ACE0 poses are only approximately metric and in an arbitrary reference frame, the script will fit a similarity transform between estimates and ground truth before calculating error.
This behaviour can be disabled with the appropriate command line flags.

## Benchmark

We evaluate the ACE0 pose quality using novel view synthesis via Nerfstudio.

**Note:** All paper results were produced with Nerfstudio v0.3.4. Since then, we updated this repository to support newer version of Nerfstudio.
We verified that benchmark results did not change significantly when updating to Nerfstudio v1.1.4. 
However, if you observe benchmarking inconsistencies w.r.t. the paper, we advise to first down-grade to Nerfstudio v0.3.4 and checkout our code using the "eccv_2024_checkpoint" git tag.

### Nerfacto

In our paper, we benchmark the ACE0 reconstruction by training a Nerfacto model and measuring PSNR on a dataset-specific training/test split of images.
To setup the benchmark, follow the instructions in the [Benchmark README](benchmarks/README.md).

Note that the benchmark lives in its own conda environment, so you have to change environments between reconstruction and benchmarking.

The benchmark takes an ACE0 pose file and fits a Nerfacto model. Optionally, you can also use our benchmarking scripts to generate the input files for Nerfstudio without running the benchmark, see the `--no_run_nerfstudio` flag.

If you do run the benchmark, it will apply a 1/8 split of the images by default to calculate PSNR.
The scripts we provide for our [Paper Experiments](#paper-experiments) do optionally run the benchmark on each dataset using the correct split.

Since the benchmarking results are stored in a nested structure, we provide a script to extract the PSNR values:

```shell
# show the benchmark results of all scenes as sub-folders in the provided top-level folder
python scripts/show_benchmark_results.py /path/to/top/level/results/folder
```

The script assumes a folder structure where each scene is a sub-folder in a dataset-specific top-level folder.
E.g. `benchmark/7scenes` contains sub-folders `chess`, `fire`, `heads`, etc.

After running benchmarking on a reconstruction, you can load the NeRF model using Nerfstudio's viewer, render videos etc.

```shell
ns-viewer --load-config /path/to/nerf/config.yaml
```

### Splatfacto

Training Gaussian splats with Splatfacto is very similar to training a Nerfacto model.
Splatfacto additionally needs a point cloud to initialise the splats, which you can export using one of our [utility scripts](#export-3d-scene-as-point-cloud) or by running ACE0 with `--export_point_cloud True`.
Our benchmarking scripts will look for a file `pc_final.ply` to pass to Nerfstudio.
Note that Nerfstudio will also proceed if no `pc_final.ply` file is found, but the splats will be initialised uniformly which can result in very poor quality.
You can check for the following warning in the Nerfstudio log to see if the point cloud was expected but missing:
```
Warning: load_3D_points set to true but no point cloud found.
```

Otherwise, you just have to run our benchmarking scripts with `--method splatfacto`, see the [Benchmark README](benchmarks/README.md) for details.
The script `show_benchmark_results.py` mentioned in the previous section also has the `--method splatfacto` option to show benchmarking metrics of Splatfacto models.

Note that all our scripts for the [Paper Experiments](#paper-experiments) support both Nerfacto and Splatfacto benchmarking.
The method can just be toggled at the top of each script. 
We recommend to take a look and use these scripts as blueprints for your own experiments.

## Paper Experiments

We provide scripts to run the main experiments of the paper.
We also provide pre-computed results for all these experiments, along with the corresponding visualizations, in the respective sections below.

### 7-Scenes

Setup the dataset.

```shell
# setup the 7-Scenes dataset in the datasets folder
cd datasets
# download and unpack the dataset
python setup_7scenes.py
# back to root directory
cd ..
```
The script can optionally convert the dataset to the ACE format, download alternative pseudo ground truth poses, calibrate depth maps, etc.
However, it is not required for the ACE0 experiments.

(Optional for the benchmark) Create a benchmarking train/test split for the 7-Scenes dataset, see the [Benchmark README](benchmarks/README.md) for details.

```shell
python scripts/create_splits_7scenes.py datasets/7scenes split_files
```

Reconstruct each scene (corresponding to "ACE0" in Table 1, left).

```shell
bash scripts/reconstruct_7scenes.sh
```

By default, the script will run with benchmarking enabled (make sure you set it up, see [Nerfacto Benchmark](#benchmark)), using Nerfacto and with visualisation disabled. 
Flip the appropriate flags in the script to change this behaviour, e.g. to train Gaussian splats instead of NeRF models. 
The ACE0 reconstruction files will be stored in `reconstructions/7scenes` while the benchmarking results will be stored in `benchmark/7scenes`.
To show the benchmarking results, call:

```shell
python scripts/show_benchmark_results.py benchmark/7scenes
```

To refine KinectFusion poses using ACE (corresponding to "KF+ACE0" in Table 1, left), run:

```shell
bash scripts/reconstruct_7scenes_warmstart.sh
# show the benchmark results
python scripts/show_benchmark_results.py benchmark/7scenes_warmstart
```

Find pre-computed poses and reconstruction videos for 7-Scenes [here](https://storage.googleapis.com/niantic-lon-static/research/acezero/results_ace0_7scenes.tar.gz). 
These results are from a different run of ACE0 than the one we used for the paper results, but PSNR values are very close (&plusmn; 0.1dB PSNR on average).

For some experiments in the paper (see right side of Table 1), we run ACE0 and baselines on a subset of images for each scene.
We provide the lists of images, together with how they have been split for the view synthesis benchmark here: [200 images per scene](https://storage.googleapis.com/niantic-lon-static/research/acezero/splits_7s_200frames.tar.gz) and [50 images per scene](https://storage.googleapis.com/niantic-lon-static/research/acezero/splits_7s_50frames.tar.gz).

### Mip-NeRF 360

Setup the dataset.

```shell
# setup the Mip-NeRF 360 dataset in the datasets folder
cd datasets
# download and unpack the dataset
python setup_mip360.py
# back to root directory
cd ..
```
The script can optionally convert the COLMAP ground truth to the ACE format, but it is not required for the ACE0 experiments.

(Optional for the benchmark) Create a benchmarking train/test split for the Mip-NeRF 360 dataset, see the [Benchmark README](benchmarks/README.md) for details. 
This uses a slightly different 1/8 split than the default benchmark split.

```shell
python scripts/create_splits_mip360.py datasets/mip360 split_files
```

Reconstruct each scene (corresponding to "ACE0" in Table 2 (b)).

```shell
bash scripts/reconstruct_mip360.sh
```

By default, the script will run with benchmarking enabled (make sure you set it up, see [Nerfacto Benchmark](#benchmark)), using Nerfacto and with visualisation disabled. 
Flip the appropriate flags in the script to change this behaviour, e.g. to train Gaussian splats instead of NeRF models.  
The ACE0 reconstruction files will be stored in `reconstructions/mip360` while the benchmarking results will be stored in `benchmark/mip360`.
To show the benchmarking results, call:

```shell
python scripts/show_benchmark_results.py benchmark/mip360
```

Find pre-computed poses and reconstruction videos for the Mip-NerF 360 dataset [here](https://storage.googleapis.com/niantic-lon-static/research/acezero/results_ace0_mip360.tar.gz). 
These results are from a different run of ACE0 than the one we used for the paper results, but PSNR values are very close (&plusmn; 0.1dB PSNR on average).

### Tanks and Temples

You have to manually [download the dataset](https://www.tanksandtemples.org/download/).
Our dataset script assumes you downloaded the group archives into `datasets/t2` without unpacking them:
```
datasets/t2/training.zip
datasets/t2/training_videos.zip
datasets/t2/intermediate.zip
datasets/t2/intermediate_videos.zip
datasets/t2/advanced.zip
datasets/t2/advanced_videos.zip
```

Setup the dataset.

```shell
# setup the T&T dataset in the datasets folder
cd datasets
# unpack the dataset
python setup_t2.py
# back to root directory
cd ..
```

Optionally, the script can download and setup COLMAP ground truth poses, and convert them to the ACE format.
This is required for the ACE0 experiments which reconstruct the dataset videos starting from a sparse COLMAP reconstruction.
Call the script with `--with-colmap`.
This will create an additional folder `t2_colmap` in the datasets folder where each scene folder not only has the image 
files, but also corresponding `*_pose.txt` files with COLMAP poses as 4x4, camera-to-world transformations.
Also, per scene, a single `focal_length.txt` file is created with the COLMAP focal length estimate.

We provide scripts for Tanks and Temples separated by scene group, i.e. training, intermediate, and advanced.
The following explanations are for the training group, but the scripts for the intermediate and advanced groups are similar.

Reconstruct each scene from a few hundred images (corresponding to "ACE0" in Table 3, left).

```shell
bash scripts/reconstruct_t2_training.sh
```

By default, the script will run with benchmarking enabled (make sure you set it up, see [Nerfacto Benchmark](#benchmark)), using Nerfacto and with visualisation disabled. 
Flip the appropriate flags in the script to change this behaviour, e.g. to train Gaussian splats instead of NeRF models. 
The ACE0 reconstruction files will be stored in `reconstructions/t2_training` while the benchmarking results will be stored in `benchmark/t2_training`.
To show the benchmarking results, call:

```shell
python scripts/show_benchmark_results.py benchmark/t2_training
```

Note that no benchmarking split files need to be generated for Tanks and Temples. 
The benchmark will apply a default 1/8 split.

To reconstruct the full videos of each scene (corresponding to "ACE0" in Table 3, right), call:
```shell
bash scripts/reconstruct_t2_training_videos.sh
# show benchmarking results
python scripts/show_benchmark_results.py benchmark/t2_training_videos
```

To reconstruct the full videos of each scene starting from a COLMAP reconstruction (corresponding to "Sparse COLMAP + ACE0" in Table 3, left), call:
```shell
bash scripts/reconstruct_t2_training_videos_warmstart.sh
# show benchmarking results
python scripts/show_benchmark_results.py benchmark/t2_training_videos_warmstart
```

Note that the last experiment assumes that you set up the dataset with `--with-colmap`.
The code will first call ACE mapping on the images with COLMAP poses to create an initial scene model.
This model is then passed to ACE0 which will use it as a seed for the full video reconstruction.
In this example, we trust the focal length estimate of COLMAP and keep it fixed throughout the reconstruction.

Find pre-computed poses and reconstruction videos for Tanks and Temples here: [Training scenes](https://storage.googleapis.com/niantic-lon-static/research/acezero/results_ace0_t2_training.tar.gz), [Intermediate scenes](https://storage.googleapis.com/niantic-lon-static/research/acezero/results_ace0_t2_intermediate.tar.gz), [Advanced scenes](https://storage.googleapis.com/niantic-lon-static/research/acezero/results_ace0_t2_advanced.tar.gz). 
These results are from a different run of ACE0 than the one we used for the paper results, but PSNR values are very close (&plusmn; 0.3dB PSNR on average).

## Frequently Asked Questions

**Q: I want Gaussian splats from my images. What do I need to do?**

Prepare ACE0 as explained in the beginning of this document: Create the ACE0 environment, compile the DSAC* bindings, and [setup Nerfstudio](benchmarks/README.md).
Then, run the following commands on your image set:

```
# activate our conda environment
conda activate ace0

# run ACE0 reconstruction with point cloud export
python ace_zero.py "/path/to/some/images/*.jpg" result_folder --export_point_cloud True

# switch to the Nerfstudio conda environment
conda activate nerfstudio

# convert the ACE0 output to a Nerfstudio compatible format and run Splatfacto training (also runs evaluation but it's fast)
python -m benchmarks.benchmark_poses --pose_file result_folder/poses_final.txt --output_dir benchmark_folder --images_glob_pattern "/path/to/some/images/*.jpg" --method splatfacto 

# view the Gaussian splats
ns-viewer --load-config benchmark_folder/nerf_data/nerf_for_eval/splatfacto/run/config.yaml
```

**Q: I run out of GPU memory during the ACE0 reconstruction. What can I do?**

**A:** All experiments in the paper were performed with 16GB of GPU memory (e.g. NVIDIA V100/T4) and the default settings should work with such a GPU.
The bulk of the memory is used by the ACE training buffer (up to ~8GB). 
You can run ACE0 with the flag `--training_buffer_cpu True` to keep the training buffer on the CPU at the expense of reconstruction speed.
With that option, ACE0 should require ~1GB of GPU memory.

**Q: I have an image collection with various images sizes, aspect ratios and intrinsics. Can I use ACE0?**

**A:** No. ACE0 assumes that all images share their intrinsics, particularly the focal length.
This is a limitation of the current implementation, rather than the method. 
Supporting images with varying intrinics should work, but would require some implementation effort, particularly in `refine_calibration.py`. 

**Q: Does ACE0 estimate intrinsics other than the focal length?**

**A:** No. ACE0 assumes that the principal point is at the image center, and pixels are square and unskewed.
The focal length, shared by all images, is the only intrinsic parameter estimated and/or refined by ACE0.

**Q: I have images from a complex camera model. e.g. with severe image distortion. Can I use ACE0?**

**A:** No. The scene coordinate regression network might be able to remove some distortion, but presumably not much.
The reprojection loss of ACE and the RANSAC pose estimator assume a pinhole camera model. 
These parts would need to implement a camera distortion model. 
If the distortion parameters are known, we would recommend to undistort the images before passing them to ACE0.

**Q: How can I run ACE0 with depth other than ZoeDepth estimates?**

**A:** If you have pre-calculated depth maps, you can call `ace_zero.py` with `--depth_files "/path/to/depths/*.png"`.
In this case, ACE0 will use the provided depth maps for the seed images instead of estimating depth.
Otherwise, the functions `get_depth_model()` and `estimate_depth()` in `dataset_io.py` can be adapted to use a depth estimator other than ZoeDepth.
Note that we found the impact of the depth estimation model to be rather small in our experiments.

**Q: Is ACE0 able to reconstruct from a small set of sparse views?**

**A:** It can work but this scenario is challenging for ACE0. 
We expect other methods, and even COLMAP, to work much better in this case.
ACE0 relies on images having sufficient visual overlap, particularly when registering new images to the reconstruction.
You can lower the registration threshold when running `ace_zero.py` via `--registration_confidence` setting it to 300 or 100 - but at some point ACE0 will get unstable.
ACE0 shines if you have dense coverage of a scene, and reconstruct it from many images in reasonable time.

## Publications

If you use ACE0 or parts of its code in your own work, please cite:

```
@inproceedings{brachmann2024acezero,
    title={Scene Coordinate Reconstruction: Posing of Image Collections via Incremental Learning of a Relocalizer},
    author={Brachmann, Eric and Wynn, Jamie and Chen, Shuai and Cavallari, Tommaso and Monszpart, {\'{A}}ron and Turmukhambetov, Daniyar and Prisacariu, Victor Adrian},
    booktitle={ECCV},
    year={2024},
}
```

This code builds on the ACE relocalizer and uses the DSAC* pose estimator. Please consider citing:

```
@inproceedings{brachmann2023ace,
    title={Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses},
    author={Brachmann, Eric and Cavallari, Tommaso and Prisacariu, Victor Adrian},
    booktitle={CVPR},
    year={2023},
}

@article{brachmann2021dsacstar,
  title={Visual Camera Re-Localization from {RGB} and {RGB-D} Images Using {DSAC}},
  author={Brachmann, Eric and Rother, Carsten},
  journal={TPAMI},
  year={2021}
}
```

ACE0 estimates depth of seed images using ZoeDepth. Please consider citing:

```
@article{bhat2023zoedepth,
  title={Zoe{D}epth: Zero-shot transfer by combining relative and metric depth},
  author={Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and M{\"u}ller, Matthias},
  journal={arXiv},
  year={2023}
}
```

This repository relies on Nerfstudio for benchmarking. 
Please consider citing according to [their docs](https://docs.nerf.studio/#citation).


## License

Copyright © Niantic, Inc. 2024. Patent Pending.
All rights reserved.
Please see the [license file](LICENSE) for terms.