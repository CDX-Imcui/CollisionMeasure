import os
import logging
import shutil
import subprocess
import sys


class Colmap:
    def __init__(self, source_path, colmap_cmd=None):
        self.source = source_path
        if colmap_cmd is None:
            self.colmap = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "colmap-x64-windows-cuda", "bin",
                                       "colmap.exe")
        else:
            self.colmap="colmap"
            # self.colmap = colmap_cmd
        self.gpu = 1
        # Paths
        self.image_dir = os.path.join(self.source, "input")
        self.database = os.path.join(self.source, "distorted/database.db")
        self.sparse_dir = os.path.join(self.source, "distorted/sparse")
        self.sparse0 = os.path.join(self.sparse_dir, '0')
        self.point_cloud = os.path.join(self.source, 'point_cloud.ply')
        self._prepare_directories()

    def _prepare_directories(self):
        os.makedirs(os.path.dirname(self.database), exist_ok=True)
        os.makedirs(os.path.join(self.source, 'distorted/sparse'), exist_ok=True)
        os.makedirs(os.path.join(self.source, 'distorted/sparse/0'), exist_ok=True)
        os.makedirs(self.source + "/sparse/0", exist_ok=True)

    def _run_cmd(self, cmd, error_msg):
        logging.info(f"Running: {cmd}")
        result = subprocess.call(cmd, shell=True)
        if result != 0:
            logging.error(error_msg)
            print("异常退出",error_msg)
            raise RuntimeError(error_msg)

    def feature_extraction(self):
        cmd = (
            f"{self.colmap} feature_extractor "
            f"--database_path {self.database} "
            f"--image_path {self.image_dir} "
            f"--ImageReader.single_camera 1 "
            f"--ImageReader.camera_model OPENCV "
            f"--SiftExtraction.use_gpu {self.gpu}"
        )
        self._run_cmd(cmd, "Feature extraction failed")

    def feature_matching(self):
        cmd = (
            f"{self.colmap} exhaustive_matcher "
            f"--database_path {self.database} "
            f"--SiftMatching.use_gpu {self.gpu}"
        )
        self._run_cmd(cmd, "Feature matching failed")

    def sparse_reconstruction(self):
        os.makedirs(self.sparse_dir, exist_ok=True)
        cmd = (
            f"{self.colmap} mapper "
            f"--database_path {self.database} "
            f"--image_path {self.image_dir} "
            f"--output_path {self.sparse_dir} "
            f"--Mapper.ba_global_function_tolerance=0.000001"
        )
        self._run_cmd(cmd, "Sparse reconstruction failed")

    def image_undistortion(self):
        # undistort into output and copy sparse files
        cmd = (
            f"{self.colmap} image_undistorter "
            f"--image_path {self.image_dir} "
            f"--input_path {self.sparse0} "
            f"--output_path {self.source} "
            f"--output_type COLMAP"
        )
        self._run_cmd(cmd, "Image undistortion failed")
        os.makedirs(self.source + "/sparse/0", exist_ok=True)
        # copy sparse files
        sparse_src = os.path.join(self.source, "sparse")
        for fname in os.listdir(sparse_src):
            if fname == '0':
                continue
            shutil.copy2(os.path.join(sparse_src, fname), os.path.join(self.sparse0, fname))

    def dense_stereo(self):
        cmd = (
            f"{self.colmap} patch_match_stereo "
            f"--workspace_path {self.source} "
            f"--workspace_format COLMAP "
            f"--PatchMatchStereo.geom_consistency true"
        )
        self._run_cmd(cmd, "Stereo matching failed")

    def stereo_fusion(self):
        cmd = (
            f"{self.colmap} stereo_fusion "
            f"--workspace_path {self.source} "
            f"--workspace_format COLMAP "
            f"--input_type geometric "
            f"--output_path {self.point_cloud}"
        )
        self._run_cmd(cmd, "Stereo fusion failed")

    def run(self):
        try:
            self.feature_extraction()
            self.feature_matching()
            self.sparse_reconstruction()
            self.image_undistortion()
            self.dense_stereo()
            self.stereo_fusion()
            logging.info("COLMAP reconstruction completed successfully.")
            print("COLMAP reconstruction completed successfully.")
        except Exception as e:
            logging.error(f"Reconstruction pipeline failed: {e}")
            raise

# colmap_command = "colmap"
# use_gpu = 1
#
# source_path = "./car"
# image_path = os.path.join(source_path, "input")
# database_path = os.path.join(source_path, "distorted/database.db")
# os.makedirs(os.path.dirname(database_path), exist_ok=True)
#
# # Feature Extraction
# cmd = f"{colmap_command} feature_extractor --database_path {database_path} --image_path {image_path} --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV  --SiftExtraction.use_gpu {use_gpu}"
# if os.system(cmd) != 0:
#     logging.error("Feature extraction failed")
#     exit(1)
#
# # Feature Matching
# cmd = f"{colmap_command} exhaustive_matcher --database_path {database_path} --SiftMatching.use_gpu {use_gpu}"
# if os.system(cmd) != 0:
#     logging.error("Feature matching failed")
#     exit(1)
#
# ### Bundle adjustment
# # The default Mapper tolerance is unnecessarily large,
# # decreasing it speeds up bundle adjustment steps.
# #   Sparse Reconstruction (Mapping)
# os.makedirs(os.path.join(source_path, 'distorted/sparse'), exist_ok=True)
# cmd = f"{colmap_command} mapper --database_path {database_path} --image_path {image_path} --output_path {os.path.join(source_path, 'distorted/sparse')} --Mapper.ba_global_function_tolerance=0.000001"
# if os.system(cmd) != 0:
#     logging.error("Sparse reconstruction failed")
#     exit(1)
#
# ### Image undistortion
# ## We need to undistort our images into ideal pinhole intrinsics.
# os.makedirs(os.path.join(source_path, 'distorted/sparse/0'), exist_ok=True)
# cmd = f"{colmap_command} image_undistorter --image_path {image_path} --input_path {os.path.join(source_path, 'distorted/sparse/0')} --output_path {source_path} --output_type COLMAP"
# if os.system(cmd) != 0:
#     logging.error("Image undistortion failed")
#     exit(1)
#
# files = os.listdir(source_path + "/sparse")
# os.makedirs(source_path + "/sparse/0", exist_ok=True)
# # Copy each file from the source directory to the destination directory
# for file in files:
#     if file == '0':
#         continue
#     source_file = os.path.join(source_path, "sparse", file)
#     destination_file = os.path.join(source_path, "sparse", "0", file)
#     shutil.copy2(source_file, destination_file)  # 使用copy2保留元数据
#     # shutil.move(source_file, destination_file)
#
# # Dense Stereo Matching
# cmd = f"{colmap_command} patch_match_stereo --workspace_path {source_path} --workspace_format COLMAP --PatchMatchStereo.geom_consistency true"
# if os.system(cmd) != 0:
#     logging.error("Stereo matching failed")
#     exit(1)
#
# # Stereo Fusion
# cmd = f"{colmap_command} stereo_fusion --workspace_path {source_path} --workspace_format COLMAP --input_type geometric --output_path {os.path.join(source_path, 'point_cloud.ply')}"
# if os.system(cmd) != 0:
#     logging.error("Stereo fusion failed")
#     exit(1)
# print("COLMAP reconstruction completed successfully.")
###################################################################


# # Poisson Meshing
# cmd = f"{colmap_command} poisson_mesher --input_path {os.path.join(dense_path, 'fused.ply')} --output_path {os.path.join(dense_path, 'meshed-poisson.ply')}"
# os.system(cmd)
#
# # Delaunay Meshing
# cmd = f"{colmap_command} delaunay_mesher --input_path {dense_path} --output_path {os.path.join(dense_path, 'meshed-delaunay.ply')}"
# os.system(cmd)

# cmd = f"{colmap_command} automatic_reconstructor  --workspace_path  {workspace_path} --image_path {images_path}"
# if os.system(cmd) != 0:
#     logging.error("automatic_reconstructor failed")
#     exit(1)
