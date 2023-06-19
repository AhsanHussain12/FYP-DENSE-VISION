import pytorch_lightning as pl
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import glob as glob
import os

from model.model import CSRNet
from utils.dataset import Sha, collate_fn

test_dir = 'Shanghai/part_A_final/test_data/'

# Load images and corresponding density maps
test_image_files = sorted(glob.glob(os.path.join(test_dir,'images', '*.jpg')))
test_density_map_files = sorted(glob.glob(os.path.join(test_dir,'densities', '*.h5')))

print(f"Lenght of test images: {len(test_image_files)}")
print(f"Lenght of test density maps: {len(test_density_map_files)}")

# create a dataset
test_images = [Image.open(img_file).convert('RGB') for img_file in test_image_files]
test_density_maps = [h5py.File(dm_file, 'r') for dm_file in test_density_map_files]
test_density_maps = [np.asarray(density_map['density']) for density_map in test_density_maps]

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
])

mae = 0
loss = nn.L1Loss()

model = CSRNet(learning_rate=1e-7)
model.load_from_checkpoint('lightning_logs/version_0/checkpoints/model-epoch=56-val_loss=6.23.ckpt', learning_rate=1e-7)

model.eval()


for i in tqdm(range(len(test_images))):
    test_image = test_images[i]
    test_density_map = test_density_maps[i]
    test_image = test_transform(test_image).unsqueeze(0)
    # resize the density map to 1/8 the size of the original image
    test_density_map = cv2.resize(test_density_map, (int(test_image.shape[3]/8), int(test_image.shape[2]/8))) * 64
    pred_density_map = model(test_image).squeeze().detach().numpy()

    # get the total number of people in the image
    pred_count = np.sum(pred_density_map)
    gt_count = np.sum(test_density_map)

    # calculate the MAE
    mae+=abs(gt_count - pred_count)



# print the final MAE in a pretty formatted table
print(f"-----\nFinal MAE: {mae/len(test_images)}\n-----")

