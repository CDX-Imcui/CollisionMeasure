from colmap_pipeline import Colmap
from argparse import ArgumentParser

parser = ArgumentParser("Colmap converter")
parser.add_argument("--source_path", "-s", required=True, type=str)
args = parser.parse_args()

colmap = Colmap(args.source_path)
colmap.run()
