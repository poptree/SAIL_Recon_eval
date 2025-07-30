run_benchmark=false
render_visualization=false
use_ba=true
benchmarking_environment="nerfstudio"
benchmarking_method="nerfacto"
# benchmarking_split_folder="benchmark_datasets/mip360_split_files"
benchmarking_out_dir="benchmark_output/vggt_ace0/tnt_training_videos"
out_dir="reconstructions/tnt_video_v6v3_30k_100_300"
datasets_folder="benchmark_datasets/tnt_training_videos"
# scenes=("Barn" "Caterpillar" "Church" "Ignatius" "Meetingroom" "Truck" "Courthouse")

mapping_exe="train_ace.py"
reconstruction_exe="ace_zero.py"

render_visualization=true

scenes=("Barn"   )

for scene in ${scenes[*]}; do
    input_rgb_files="${datasets_folder}/${scene}/*.jpg"
    scene_out_dir="${out_dir}/${scene}"

    vggt_pose_file="${out_dir}/training__${scene}/pred.txt"
    intrinsics_file="${out_dir}/training__${scene}/intrinsic_row.txt"
    output_ace_file="${out_dir}/training__${scene}/ace_pose_filter.txt"
    confidence_point_file="${out_dir}/training__${scene}/confidence_xyz.txt"

    mkdir -p ${scene_out_dir}
    mkdir -p  "${scene_out_dir}/renderings"

    # Convert the VGGT pose to ACE pose
    python benchmarks/convert_vggt_to_ace_with_confidence_filter.py --pred_poses $vggt_pose_file   --data_pattern "${input_rgb_files}" --output_file $output_ace_file --intrinsics $intrinsics_file --confidence_depth_file ${confidence_point_file}
    
    # read vggt focal length
    calibration_file="${out_dir}/training__${scene}/focal_length.txt"
    focal_length=$(cat ${calibration_file})
    echo "Using focal length from vggt stage: ${focal_length}"

    if ${run_benchmark} && [ "${benchmarking_method}" = "splatfacto" ]; then
    export_pc_cmd="--export_point_cloud True --dense_point_cloud ${benchmarking_dense_pcinit}"
    else
        export_pc_cmd="--export_point_cloud False --dense_point_cloud False"
    fi

    visualization_cmd="--render_visualization ${render_visualization}"
    # network name set to a particular pattern to make sure the visualization works
    network_name="iteration0_seed0"

    # runing ace and ace0 as (pose refinement)
    python ${mapping_exe} "${input_rgb_files}" ${scene_out_dir}/${network_name}.pt --use_ace_pose_file "${output_ace_file}" ${visualization_cmd} --render_target_path "${scene_out_dir}/renderings"  2>&1 | tee ${scene_out_dir}/log_${scene}_init.txt
    # focal_length=$(echo "scale=6; ${focal_length} / 2" | bc)
    # echo "Adjusted focal length for video frames: ${focal_length}"
    python $reconstruction_exe "${input_rgb_files}" ${scene_out_dir} --seed_network ${scene_out_dir}/${network_name}.pt  ${visualization_cmd} --use_external_focal_length ${focal_length} --refine_calibration False ${export_pc_cmd} 2>&1 | tee ${scene_out_dir}/log_${scene}.txt
    if $run_benchmark; then
        benchmarking_scene_dir="${benchmarking_out_dir}/${scene}"
        mkdir -p ${benchmarking_scene_dir}
        conda run --no-capture-output -n ${benchmarking_environment} python -m benchmarks.benchmark_poses --pose_file ${scene_out_dir}/poses_final.txt --output_dir ${benchmarking_scene_dir} --images_glob_pattern "${input_rgb_files_video}" --method ${benchmarking_method} 2>&1 | tee ${benchmarking_out_dir}/log_${scene}.txt
    fi
done


#   visualization_cmd="--render_visualization ${render_visualization}"
#   if ${run_benchmark} && [ "${benchmarking_method}" = "splatfacto" ]; then
#     export_pc_cmd="--export_point_cloud True --dense_point_cloud ${benchmarking_dense_pcinit}"
#   else
#     export_pc_cmd="--export_point_cloud False --dense_point_cloud False"
#   fi