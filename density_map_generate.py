from scipy.ndimage import gaussian_filter
from scipy import spatial
import numpy as np
import cv2
import os
import glob as glob

import pandas as pd
from tqdm import tqdm

class DensityMapGenerator:
    def __init__(self, image_h=600, image_w=800):
        # Store image dimensions
        self.image_h = image_h
        self.image_w = image_w
        
    def dotmap_from_csv(self, csv_file):
        # Read the CSV file using pandas
        data = pd.read_csv(csv_file)
        
        # Create dot map
        dot_map = np.zeros((self.image_h, self.image_w))
        
        # Assuming the CSV columns are named 'x' and 'y'
        for index, row in data.iterrows():
            y, x = int(row['Y']), int(row['X'])
            dot_map[y][x] = 1
        
        return dot_map
    
    def gaussian_filter_density(self, gt):
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

        for i, pt in enumerate(pts):
            pt2d = np.zeros(gt.shape, dtype=np.float32)
            pt2d[pt[1], pt[0]] = 1.
            if gt_count > 1:
                sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
            else:
                sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
            density += gaussian_filter(pt2d, sigma, mode='constant')
        return density
    
    def generate(self, csv_file):
        # Convert CSV annotations to dot map
        dot_map = self.dotmap_from_csv(csv_file)
        
        # Generate density map using the provided method
        density_map = self.gaussian_filter_density(dot_map)
        
        return density_map

if __name__ == '__main__':
    generator = DensityMapGenerator()
    all_csv_files_train = glob.glob('train/ground_truth/*.csv')
    all_csv_files_test = glob.glob('test/ground_truth/*.csv')

    # create density_maps folder if it doesn't exist
    if not os.path.exists('train/density_maps'):
        os.makedirs('train/density_maps')

    if not os.path.exists('test/density_maps'):
        os.makedirs('test/density_maps')
    


    for csv_file in tqdm(all_csv_files_train, desc='Generating density maps', unit='files',colour='green'):
        # Generate density map
        density_map = generator.generate(csv_file)
        
        # Save density map
        filename = os.path.basename(csv_file).replace('.csv', '')
        np.save('train/density_maps/' + filename + '.npy', density_map)

    for csv_file in tqdm(all_csv_files_test, desc='Generating density maps', unit='files',colour='green'):
        # Generate density map
        density_map = generator.generate(csv_file)
        
        # Save density map
        filename = os.path.basename(csv_file).replace('.csv', '')
        np.save('test/density_maps/' + filename + '.npy', density_map)

    print('Done.')