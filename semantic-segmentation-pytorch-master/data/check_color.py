import numpy as np
from scipy.io import loadmat
import csv

# Load the colors
colors = loadmat('data/color150.mat')['colors']

# Load the object names
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row
    for row in reader:
        # The object IDs in object150_info.csv are 1-indexed,
        # while array indices are 0-indexed.
        # So, if you want to map directly, you'd use (id - 1) for colors array.
        obj_id = int(row[0])
        obj_name = row[5].split(";")[0]
        names[obj_id] = obj_name

print("--- Semantic Object Color Mapping ---")
# Iterate from 1 to 150 (inclusive) for object IDs
for i in range(1, 151):
    if i in names and i - 1 < len(colors):
        object_name = names[i]
        # Access the color for this object. Remember colors array is 0-indexed.
        color_rgb = colors[i - 1]
        print(f"Object ID: {i}, Name: {object_name}, Color (RGB): {color_rgb}")
    elif i not in names:
        print(f"Warning: Object ID {i} not found in object150_info.csv")
    elif i - 1 >= len(colors):
        print(f"Warning: Color for Object ID {i} (index {i-1}) out of bounds in color150.mat")

