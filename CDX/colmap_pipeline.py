import os
import logging
from argparse import ArgumentParser
import shutil

# parser = ArgumentParser("COLMAP reconstruction pipeline")
# parser.add_argument("--no_gpu", action='store_true')
# parser.add_argument("--source_path", "-s", required=True, type=str)
# parser.add_argument("--colmap_executable", default="colmap", type=str)
# args = parser.parse_args()

# colmap_command = f'"{args.colmap_executable}"' if args.colmap_executable else "colmap"
colmap_command = "colmap"
# use_gpu = 0 if args.no_gpu else 1
use_gpu = 1

# source_path = args.source_path
source_path = "./car"
image_path = os.path.join(source_path, "input")
database_path = os.path.join(source_path, "distorted/database.db")
os.makedirs(os.path.dirname(database_path), exist_ok=True)
# sparse_path = os.path.join(source_path, "sparse")
# os.makedirs(sparse_path, exist_ok=True)
# dense_path = os.path.join(source_path, "dense")
# os.makedirs(dense_path, exist_ok=True)





# Feature Extraction
cmd = f"{colmap_command} feature_extractor --database_path {database_path} --image_path {image_path} --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV  --SiftExtraction.use_gpu {use_gpu}"
if os.system(cmd) != 0:
    logging.error("Feature extraction failed")
    exit(1)

# Feature Matching
cmd = f"{colmap_command} exhaustive_matcher --database_path {database_path} --SiftMatching.use_gpu {use_gpu}"
if os.system(cmd) != 0:
    logging.error("Feature matching failed")
    exit(1)

### Bundle adjustment
# The default Mapper tolerance is unnecessarily large,
# decreasing it speeds up bundle adjustment steps.
#   Sparse Reconstruction (Mapping)
os.makedirs(os.path.join(source_path, 'distorted/sparse'), exist_ok=True)
cmd = f"{colmap_command} mapper --database_path {database_path} --image_path {image_path} --output_path {os.path.join(source_path, 'distorted/sparse')} --Mapper.ba_global_function_tolerance=0.000001"
if os.system(cmd) != 0:
    logging.error("Sparse reconstruction failed")
    exit(1)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
os.makedirs(os.path.join(source_path, 'distorted/sparse/0'), exist_ok=True)
cmd = f"{colmap_command} image_undistorter --image_path {image_path} --input_path {os.path.join(source_path, 'distorted/sparse/0')} --output_path {source_path} --output_type COLMAP"
if os.system(cmd) != 0:
    logging.error("Image undistortion failed")
    exit(1)

files = os.listdir(source_path + "/sparse")
os.makedirs(source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(source_path, "sparse", file)
    destination_file = os.path.join(source_path, "sparse", "0", file)
    shutil.copy2(source_file, destination_file)  # 使用copy2保留元数据
    # shutil.move(source_file, destination_file)

# Dense Stereo Matching
cmd = f"{colmap_command} patch_match_stereo --workspace_path {source_path} --workspace_format COLMAP --PatchMatchStereo.geom_consistency true"
if os.system(cmd) != 0:
    logging.error("Stereo matching failed")
    exit(1)

# Stereo Fusion
cmd = f"{colmap_command} stereo_fusion --workspace_path {source_path} --workspace_format COLMAP --input_type geometric --output_path {os.path.join(source_path, 'point_cloud.ply')}"
if os.system(cmd) != 0:
    logging.error("Stereo fusion failed")
    exit(1)

# # Poisson Meshing
# cmd = f"{colmap_command} poisson_mesher --input_path {os.path.join(dense_path, 'fused.ply')} --output_path {os.path.join(dense_path, 'meshed-poisson.ply')}"
# os.system(cmd)
#
# # Delaunay Meshing
# cmd = f"{colmap_command} delaunay_mesher --input_path {dense_path} --output_path {os.path.join(dense_path, 'meshed-delaunay.ply')}"
# os.system(cmd)
print("COLMAP reconstruction completed successfully.")

# cmd = f"{colmap_command} automatic_reconstructor  --workspace_path  {workspace_path} --image_path {images_path}"
# if os.system(cmd) != 0:
#     logging.error("automatic_reconstructor failed")
#     exit(1)
