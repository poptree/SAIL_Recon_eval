run_benchmark=true
render_visualization=true
use_ba=true
baconfigs=baconfig/ba_video.json
benchmarking_environment="nerfstudio"
benchmarking_method="nerfacto"
benchmarking_out_dir="benchmark_output/sail-recon/tnt_intermediate_videos"
out_dir="reconstructions/tnt_images/"
datasets_folder="benchmark_datasets/tnt_intermediate"
scenes=("Family" "Francis" "Horse" "Lighthouse" "Playground" "Train")

for scene in ${scenes[*]}; do
    input_rgb_files="${datasets_folder}/${scene}/*.jpg"
    scene_out_dir="${out_dir}/${scene}"

    sailrecon_pose_file="${out_dir}/intermediate__${scene}/pred.txt"
    intrinsics_file="${out_dir}/intermediate__${scene}/intrinsic_row.txt"
    output_ace_file="${out_dir}/intermediate__${scene}/ace_pose.txt"

    mkdir -p ${scene_out_dir}

    # Convert the sailrecon pose to ACE pose
    python benchmarks/convert_sailrecon_to_ace.py --pred_poses $sailrecon_pose_file   --data_pattern "${input_rgb_files}" --output_file $output_ace_file --intrinsics $intrinsics_file --split_json ${benchmarking_split_folder}/7scenes_${scene}.json
    if $run_benchmark; then
        benchmarking_scene_dir="${benchmarking_out_dir}/${scene}"
        mkdir -p ${benchmarking_scene_dir}
        conda run --no-capture-output -n ${benchmarking_environment} python -m benchmarks.benchmark_poses --pose_file ${output_ace_file} --output_dir ${benchmarking_scene_dir} --images_glob_pattern "${input_rgb_files}"  --method ${benchmarking_method} --camera_optimizer off --run_ba ${baconfigs}   2>&1 | tee ${benchmarking_out_dir}/log_${scene}.txt
    fi
done
