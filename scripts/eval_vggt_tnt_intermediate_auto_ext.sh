run_benchmark=true
render_visualization=true
use_ba=true
baconfigs=baconfig/nerfacto_imageset.json
benchmarking_environment="nerfstudio"
benchmarking_method="nerfacto"
# benchmarking_split_folder="benchmark_datasets/mip360_split_files"
benchmarking_out_dir="benchmark_output/vggt_ba_1e-31e-5_nopen/tnt_intermediate"
out_dir="reconstructions/tnt_imagev6v3_30k_100_300"
datasets_folder="benchmark_datasets/tnt_intermediate"
# scenes=("Barn" "Caterpillar" "Church" "Ignatius" "Meetingroom" "Truck")
scenes=("M60" "Panther")
# scenes=( )

for scene in ${scenes[*]}; do
    input_rgb_files="${datasets_folder}/${scene}/*.jpg"
    scene_out_dir="${out_dir}/${scene}"

    vggt_pose_file="${out_dir}/intermediate__${scene}/pred.txt"
    intrinsics_file="${out_dir}/intermediate__${scene}/intrinsic_row.txt"
    output_ace_file="${out_dir}/intermediate__${scene}/ace_pose.txt"


    mkdir -p ${scene_out_dir}

    # Convert the VGGT pose to ACE pose
    python benchmarks/convert_vggt_to_ace.py --pred_poses $vggt_pose_file   --data_pattern "${input_rgb_files}" --output_file $output_ace_file --intrinsics $intrinsics_file

    if $run_benchmark; then
        benchmarking_scene_dir="${benchmarking_out_dir}/${scene}"
        mkdir -p ${benchmarking_scene_dir}
        python -m benchmarks.benchmark_poses --pose_file ${output_ace_file} --output_dir ${benchmarking_scene_dir} --images_glob_pattern "${input_rgb_files}"  --method ${benchmarking_method} --camera_optimizer off --run_ba ${baconfigs}  2>&1 | tee ${benchmarking_out_dir}/log_${scene}.txt
    fi
done


#   visualization_cmd="--render_visualization ${render_visualization}"
#   if ${run_benchmark} && [ "${benchmarking_method}" = "splatfacto" ]; then
#     export_pc_cmd="--export_point_cloud True --dense_point_cloud ${benchmarking_dense_pcinit}"
#   else
#     export_pc_cmd="--export_point_cloud False --dense_point_cloud False"
#   fi