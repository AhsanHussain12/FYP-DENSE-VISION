import os
import glob
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import scipy
from scipy.ndimage import gaussian_filter
from scipy import spatial  # Import spatial for KDTree

def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density

# Set the root to the Shanghai dataset path
root = 'D://CSRNET_Repo//Shanghai//part_A_final//test_data//images//'  # Updated to your Shanghai dataset path



img_paths = []
for img_path in glob.glob(os.path.join(root, '*.jpg')):  # Change '.png' to '.jpg'
    img_paths.append(img_path)
print("Image paths found:", img_paths)
for img_path in img_paths:
    print("IMAGE PATH-> ", img_path)
    # Adjusted for JPEG ground truth
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["image_info"][0, 0][0, 0][0]
    
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
            
    k = gaussian_filter_density(k)
    
    # Save density map in HDF5 format
    with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'densities'), 'w') as hf:  # Change '.png' to '.jpg'
        hf['density'] = k

# Now see a sample from ShanghaiA
plt.imshow(Image.open(img_paths[0]))
gt_file = h5py.File(img_paths[0].replace('.jpg', '.h5').replace('images', 'densities'), 'r')  # Change '.png' to '.jpg'
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth, cmap='jet')
np.sum(groundtruth)  # Don't mind this slight variation
