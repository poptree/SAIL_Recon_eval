run_benchmark=true
render_visualization=true
use_ba=false
baconfigs=baconfig/nerfacto_7scenes_t.json
benchmarking_environment="nerfstudio"
benchmarking_method="nerfacto"
benchmarking_split_folder="benchmark_datasets/7scenes"
benchmarking_out_dir="benchmark_output/vggt_multiba/7scenes_v6v4_30k_100_300_t"
out_dir="reconstructions/7scenes/v6v4_30k_100_300"
datasets_folder="benchmark_datasets/7scenes"
# scenes=("stairs" )

scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")

for scene in ${scenes[*]}; do
    input_rgb_files="${datasets_folder}/${scene}/seq-*/*.color.png"
    scene_out_dir="${out_dir}/${scene}"

    vggt_pose_file="${out_dir}/${scene}/pred.txt"
    intrinsics_file="${out_dir}/${scene}/intrinsic_row.txt"
    output_ace_file="${out_dir}/${scene}/ace_pose.txt"


    mkdir -p ${scene_out_dir}

    # Convert the VGGT pose to ACE pose
    python benchmarks/convert_vggt_to_ace.py --pred_poses $vggt_pose_file   --data_pattern "${input_rgb_files}" --output_file $output_ace_file --intrinsics $intrinsics_file --split_json ${benchmarking_split_folder}/7scenes_${scene}.json 

    if $run_benchmark; then
        benchmarking_scene_dir="${benchmarking_out_dir}/${scene}"
        mkdir -p ${benchmarking_scene_dir}
        python -m benchmarks.benchmark_poses --pose_file ${output_ace_file} --output_dir ${benchmarking_scene_dir} --images_glob_pattern "${input_rgb_files}" --split_json ${benchmarking_split_folder}/7scenes_${scene}.json --method ${benchmarking_method} --camera_optimizer off --run_ba ${baconfigs}    2>&1 | tee ${benchmarking_out_dir}/log_${scene}.txt
          benchmarking_scene_dir="${benchmarking_out_dir}/${scene}"
        mkdir -p ${benchmarking_scene_dir}
        # python -m benchmarks.benchmark_poses --pose_file ${output_ace_file} --output_dir ${benchmarking_scene_dir} --images_glob_pattern "${input_rgb_files}" --split_json ${benchmarking_split_folder}/7scenes_${scene}.json --method ${benchmarking_method} --camera_optimizer off   2>&1 | tee ${benchmarking_out_dir}/log_${scene}.txt
    fi
done



# evo_ape kitti /mnt/jfs/hengli/projects/vggt_workspace/acezero/reconstructions/7scenes/office/gt.txt /mnt/jfs/hengli/projects/vggt_workspace/acezero/reconstructions/7scenes/stairs/pred.txt  -vas --save_plot /mnt/jfs/hengli/projects/vggt_workspace/acezero/reconstructions/7scenes/stairs/

# evo_ape kitti reconstructions/7scenes/v6v4_30k_100_300/fire/gt.txt   reconstructions/7scenes/v6v4_30k_100_300/fire/pred.txt  -vas --save_plot  reconstructions/7scenes/v6v4_30k_100_300/fire/
# reconstructions/7scenes_test_30k_50_150_300/pumpkin/pred
# evo_ape kitti /mnt/jfs/hengli/projects/vggt_workspace/acezero/reconstructions/7scenes_test_30k_50_150_300/pumpkin/gt.txt /mnt/jfs/hengli/projects/vggt_workspace/acezero/reconstructions/7scenes_test_30k_50_150_300/pumpkin/pred_ba.txt  -vas --save_plot /mnt/jfs/hengli/projects/vggt_workspace/acezero/reconstructions/7scenes_test_30k_50_150_300/pumpkin/ba
# python benchmarks/ransac_eval_pose.py --gt_poses /mnt/jfs/hengli/projects/vggt_workspace/acezero/reconstructions/7scenes_test_30k_50_150_300/pumpkin/gt.txt --aligned_poses /mnt/jfs/hengli/projects/vggt_workspace/acezero/reconstructions/7scenes_test_30k_50_150_300/pumpkin/pred_ba.txt --save_dir /mnt/jfs/hengli/projects/vggt_workspace/acezero/reconstructions/7scenes_test_30k_50_150_300/pumpkin/results_ba.txt

# python benchmarks/ransac_eval_pose.py --gt_poses /mnt/jfs/hengli/projects/vggt_workspace/acezero/reconstructions/7scenes_test_30k_50_150_300/pumpkin/gt.txt --aligned_poses /mnt/jfs/hengli/projects/vggt_workspace/acezero/reconstructions/7scenes_test_30k_50_150_300/pumpkin/pred.txt --save_dir /mnt/jfs/hengli/projects/vggt_workspace/acezero/reconstructions/7scenes_test_30k_50_150_300/pumpkin/results.txt
#   visualization_cmd="--render_visualization ${render_visualization}"
#   if ${run_benchmark} && [ "${benchmarking_method}" = "splatfacto" ]; then
#     export_pc_cmd="--export_point_cloud True --dense_point_cloud ${benchmarking_dense_pcinit}"
#   else
#     export_pc_cmd="--export_point_cloud False --dense_point_cloud False"
#   fi