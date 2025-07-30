python -m benchmarks.benchmark_poses  --pose_file ./data/bicycle/ace_pose.txt --output_dir ./data/bicycle/benchmark_barf/ --images_glob_pattern "benchmark_datasets/mip360/bicycle/images_4/*JPG" --split_json ./data/mip360_bicycle.json --camera_optimizer SE3  2>&1 | tee data/bicycle/benchmark/log_bicycle.txt


python -m benchmarks.benchmark_poses  --pose_file ./data/bicycle/ace_pose.txt --output_dir ./data/bicycle/benchmark_barf/ --images_glob_pattern "benchmark_datasets/mip360/bicycle/images_4/*JPG" --split_json ./data/mip360_bicycle_barf.json --camera_optimizer SO3xR3  2>&1 | tee data/bicycle/benchmark_barf/log_bicycle.txt


python -m benchmarks.benchmark_poses  --pose_file ./data/bicycle/ace_pose.txt --output_dir ./data/bicycle/benchmark_ba_new/ --images_glob_pattern "benchmark_datasets/mip360/bicycle/images_4/*JPG" --split_json ./data/mip360_bicycle.json --camera_optimizer off --run_ba  2>&1 | tee data/bicycle/benchmark_ba_new/log_bicycle.txt


python -m benchmarks.benchmark_poses  --pose_file ./data/bonsai/ace_pose.txt --output_dir ./data/bonsai/benchmark_ba/ --images_glob_pattern "benchmark_datasets/mip360/bonsai/images_4/*JPG" --split_json ./data/mip360_bonsai.json --camera_optimizer off --run_ba  2>&1 | tee data/bonsai/benchmark_ba/log_bicycle.txt

python -m benchmarks.benchmark_poses  --pose_file ./data/counter/ace_pose.txt --output_dir ./data/counter/benchmark_ba/ --images_glob_pattern "benchmark_datasets/mip360/counter/images_4/*JPG" --split_json ./data/mip360_counter.json --camera_optimizer off --run_ba  2>&1 | tee data/counter/benchmark_ba/log_bicycle.txt

python -m benchmarks.benchmark_poses  --pose_file ./data/garden/ace_pose.txt --output_dir ./data/garden/benchmark_ba_15000/ --images_glob_pattern "benchmark_datasets/mip360/garden/images_4/*JPG" --split_json ./data/mip360_garden.json --camera_optimizer off --run_ba  2>&1 | tee data/garden/benchmark_ba_15000/log_bicycle.txt


python -m benchmarks.benchmark_poses  --pose_file ./data/kitchen/ace_pose.txt --output_dir ./data/kitchen/benchmark_ba/ --images_glob_pattern "benchmark_datasets/mip360/kitchen/images_4/*JPG" --split_json ./data/mip360_kitchen.json --camera_optimizer off --run_ba  2>&1 | tee data/kitchen/benchmark_ba/log_bicycle.txt


python -m benchmarks.benchmark_poses  --pose_file ./data/room/ace_pose.txt --output_dir ./data/room/benchmark_ba/ --images_glob_pattern "benchmark_datasets/mip360/room/images_4/*JPG" --split_json ./data/mip360_room.json --camera_optimizer off --run_ba  2>&1 | tee data/room/benchmark_ba/log_bicycle.txt

python -m benchmarks.benchmark_poses  --pose_file ./data/stump/ace_pose.txt --output_dir ./data/stump/benchmark_ba/ --images_glob_pattern "benchmark_datasets/mip360/stump/images_4/*JPG" --split_json ./data/mip360_stump.json --camera_optimizer off --run_ba  2>&1 | tee data/stump/benchmark_ba/log_bicycle.txt