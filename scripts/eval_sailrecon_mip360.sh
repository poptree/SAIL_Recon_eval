run_benchmark=true
render_visualization=true
use_ba=true
baconfigs=baconfig/ba_imageset.json
benchmarking_environment="nerfstudio"
benchmarking_method="nerfacto"
benchmarking_split_folder="SPLIT"
benchmarking_out_dir="benchmark_output/sailrecon/mip360/"
out_dir="reconstructions/mip360/"
datasets_folder="benchmark_datasets/mip360"
scenes=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump")

for scene in ${scenes[*]}; do
    input_rgb_files="${datasets_folder}/${scene}/images_4/*.JPG"
    scene_out_dir="${out_dir}/${scene}"

    sailrecon_pose_file="${out_dir}/${scene}/pred.txt"
    intrinsics_file="${out_dir}/${scene}/intrinsic_row.txt"
    output_ace_file="${out_dir}/${scene}/ace_pose.txt"


    mkdir -p ${scene_out_dir}

    # Convert the sailrecon pose to ACE pose
    python benchmarks/convert_sailrecon_to_ace.py --pred_poses $sailrecon_pose_file   --data_pattern "${input_rgb_files}" --output_file $output_ace_file --intrinsics $intrinsics_file

    if $run_benchmark; then
        benchmarking_scene_dir="${benchmarking_out_dir}/${scene}"
        mkdir -p ${benchmarking_scene_dir}
        conda run --no-capture-output -n ${benchmarking_environment} python -m benchmarks.benchmark_poses --pose_file ${output_ace_file} --output_dir ${benchmarking_scene_dir} --images_glob_pattern "${input_rgb_files}" --split_json ${benchmarking_split_folder}/mip360_${scene}.json --method ${benchmarking_method} --camera_optimizer off --run_ba ${baconfigs}  2>&1 | tee ${benchmarking_out_dir}/log_${scene}.txt
    fi
done

