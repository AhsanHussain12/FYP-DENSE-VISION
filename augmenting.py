import argparse
import glob
import json
import logging
import os
import random
import sys
import time

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from utils.augmentation import create_dataloader
from utils.model import CSRNet

# AC-AL augmentation

# Load labeled dataset
# replace with your actual paths
images_dir = 'Shanghai/part_A_final/train_data/images'
density_maps_dir = 'Shanghai/part_A_final/train_data/densities'

# Load images and corresponding density maps
image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
density_map_files = sorted(glob.glob(os.path.join(density_maps_dir, '*.h5')))

# select 10 random images and their corresponding density maps

image_files = random.sample(image_files, 10)
image_files = ['Shanghai/part_A_final/train_data/images/IMG_132.jpg',
'Shanghai/part_A_final/train_data/images/IMG_83.jpg',
'Shanghai/part_A_final/train_data/images/IMG_248.jpg',
'Shanghai/part_A_final/train_data/images/IMG_281.jpg',
'Shanghai/part_A_final/train_data/images/IMG_170.jpg',
'Shanghai/part_A_final/train_data/images/IMG_230.jpg',
'Shanghai/part_A_final/train_data/images/IMG_41.jpg',
'Shanghai/part_A_final/train_data/images/IMG_263.jpg',
'Shanghai/part_A_final/train_data/images/IMG_154.jpg',
'Shanghai/part_A_final/train_data/images/IMG_124.jpg']
density_map_files = [f.replace('images', 'densities').replace('jpg','h5') for f in image_files]


images = [Image.open(img_file).convert('RGB') for img_file in image_files]
density_maps = [h5py.File(dm_file, 'r') for dm_file in density_map_files]
density_maps = [np.asarray(density_map['density']) for density_map in density_maps]

def numpy_crop(np_array, i, j, h, w):
    """Crop a numpy array."""
    return np_array[i:i+h, j:j+w]

def augment_images(images, density_maps, ref_number):
    augmented_images = []
    augmented_density_maps = []
    num_patches_per_image = ref_number // len(images)
    for image, density_map in zip(images, density_maps):
        for _ in range(num_patches_per_image):
            image = np.asarray(image)

            # Calculate the crop size for this image
            crop_size = (image.shape[0] // 4, image.shape[1] // 4)

            # Apply the same random crop to the image and its density map
            
            rand_i,rand_j,rand_h,rand_w = random.randint(0, image.shape[0] - crop_size[0]), random.randint(0, image.shape[1] - crop_size[1]), crop_size[0], crop_size[1]

            
            augmented_image = numpy_crop(image, rand_i, rand_j, rand_h, rand_w)
            augmented_density_map = numpy_crop(density_map, rand_i, rand_j, rand_h, rand_w)


            # i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
            # augmented_image = transforms.functional.crop(image, i, j, h, w)
            # augmented_density_map = numpy_crop(density_map, i, j, h, w)
            
            assert augmented_image.shape[:2] == augmented_density_map.shape[:2]

            augmented_images.append(augmented_image)
            augmented_density_maps.append(augmented_density_map)
    return augmented_images, augmented_density_maps

def create_minibatch(images, density_maps):
    for i in range(0, len(images), 2):
        # get the minimum dimension across both images
        min_height = min(images[i].shape[0], images[i+1].shape[0])
        min_width = min(images[i].shape[1], images[i+1].shape[1])


        # crop the images and density maps to the minimum dimension 
        image1 = images[i][:min_height, :min_width]
        image2 = images[i+1][:min_height, :min_width]

        density_map1 = density_maps[i][:min_height, :min_width]
        density_map2 = density_maps[i+1][:min_height, :min_width]

        # resize the density maps to 1/8 the size of the original image
        density_map1 = cv2.resize(density_map1, (int(min_width/8), int(min_height/8))) * 64
        density_map2 = cv2.resize(density_map2, (int(min_width/8), int(min_height/8))) * 64
    

        yield (image1, image2, density_map1, density_map2)
        

        # i, j, h, w = transforms.RandomCrop.get_params(images[i], output_size=(min_height, min_width))
        # yield (transforms.functional.crop(images[i], i, j, h, w), 
        #        transforms.functional.crop(images[i+1], i, j, h, w), 
        #        numpy_crop(density_maps[i], i, j, h, w), 
        #        numpy_crop(density_maps[i+1], i, j, h, w))

reference_number = 1200  # desired number of patches per dataset
augmented_images, augmented_density_maps = augment_images(images, density_maps, reference_number)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MOMENTUM = 0.95
DECAY = 5e-4
LR = 1e-6
model = CSRNet(training=True).to(device)
criterion = nn.MSELoss(reduction='sum')
optim = torch.optim.SGD(
        model.parameters(), LR, momentum=MOMENTUM, weight_decay=DECAY)


model.train()
train_loss = 0.0
epochs = 100
mae = torch.nn.L1Loss(reduction='sum')
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])

cols = [f"Epoch_{i}" for i in range(epochs)]
loss_history = pd.DataFrame(columns=cols)

best_loss = 1e6
for epoch in range(epochs):
    model.train()
    # Shuffle data at the beginning of each epoch
    indices = torch.randperm(len(augmented_images))
    augmented_images = [augmented_images[i] for i in indices]
    augmented_density_maps = [augmented_density_maps[i] for i in indices]
    
    minibatches = list(create_minibatch(augmented_images, augmented_density_maps))

    loss = torch.tensor(0.0)
    loss_info = []
    
    with tqdm(minibatches) as tepochs:
        for image1,image2,density1,density2 in tepochs:



            image1 = train_transforms(image1)
            image2 = train_transforms(image2)

            images = torch.stack([image1,image2],dim=0)
            images = images.to(device)

            image1 = image1.to(device)
            image2 = image2.to(device)


            densities = torch.stack([torch.from_numpy(density1).unsqueeze(0),torch.from_numpy(density2).unsqueeze(0)],dim=0)
            densities = densities.to(device)

            # print(f"Images size: {images.shape}")
            # print(f"Densities size: {densities.shape}")

            optim.zero_grad()
            outputs = model(images)
            # print(f"Outputs size: {outputs.shape}")
            loss = criterion(outputs, densities)
            loss.backward()

            optim.step()

            tepochs.set_postfix(loss=loss.item())
        
    model.eval()
    with torch.no_grad():
        for i in tqdm(image_files):
            image = Image.open(i).convert('RGB')
            image = train_transforms(image)
            image = image.unsqueeze(0)
            image = image.to(device)
            output = model(image)
            
            loss = mae(output.sum(),torch.from_numpy(density_maps[0]).sum())
            loss_info.append(loss.item())
    print(f"Validation MAE: {np.mean(loss_info)}")

    if np.mean(loss_info) < best_loss:
        print(f"New best loss. Saving model..")
        best_loss = np.mean(loss_info)
        torch.save(model.state_dict(), 'best_model.pth')
        
    with open('loss_info.txt', 'a') as f:
        f.write(f"{epoch},{np.mean(loss_info)}\n")
        
    print(f"Loss: {loss.item()}")
    print(f"Epoch: {epoch}/{epochs}")

    print(f"Inference to save predictions")
    model.eval()
    with torch.no_grad():
        for i in tqdm(image_files):
            image = Image.open(i).convert('RGB')
            image = train_transforms(image)
            image = image.unsqueeze(0)
            image = image.to(device)
            output = model(image)
            
            loss_history.loc[i,f"Epoch_{epoch}"] = output.sum().item()



loss_history.to_csv("loss_history.csv")

        

       

        

