import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py
from scipy.ndimage import gaussian_filter
from scipy import spatial
import scipy.io as io

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)

    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    tree = spatial.KDTree(pts.copy(), leafsize=2048)
    distances, _ = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.0
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 4.
        density += gaussian_filter(pt2d, sigma, mode='constant')
    return density

# === Set folder paths ===
image_folder = 'D://Ashhad//FYP//Dataset//Challenging_Env_Dataset//Train//image'
gt_folder = 'D://Ashhad//FYP//Dataset//Challenging_Env_Dataset//Train//ground_truth'
density_output_folder = 'D://Ashhad//FYP//Dataset//Challenging_Env_Dataset//Train//density_map'

# Create output folder if not exists
os.makedirs(density_output_folder, exist_ok=True)

# === Process all .jpg images ===
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

print(f"Found {len(image_files)} image(s) to process.\n")

for idx, image_file in enumerate(sorted(image_files), 1):
    print(f"Processing [{idx}/{len(image_files)}]: {image_file}")

    # Paths
    img_path = os.path.join(image_folder, image_file)
    mat_file = image_file.replace('.jpg', '.mat')
    mat_path = os.path.join(gt_folder, mat_file)
    h5_path = os.path.join(density_output_folder, image_file.replace('.jpg', '.h5'))

    # Check if ground truth file exists
    if not os.path.exists(mat_path):
        print(f"⚠️  Skipping {image_file} — No matching .mat file found.")
        continue

    # Load data
    mat = io.loadmat(mat_path)
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]))

    gt = mat['image_info'][0, 0][0][0]
    for i in range(len(gt)):
        x, y = int(gt[i][0]), int(gt[i][1])
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            k[y, x] = 1

    # Generate density map
    density_map = gaussian_filter_density(k)

    # Save to HDF5
    with h5py.File(h5_path, 'w') as hf:
        hf['density'] = density_map

    print(f"✅ Saved density map to: {h5_path}")
    print(f"   Estimated count: {np.sum(density_map):.2f}\n")

print("\n🎉 All images processed.")
