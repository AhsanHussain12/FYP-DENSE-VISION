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

from model.model import CSRNet
from utils.dataset import Sha, collate_fn


def main():
    
    # DataLoader creation
    # Assuming images and density_maps are lists of numpy arrays
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
    ref_number = 1200

    train_transforms = transforms.Compose([
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
    ])

    dataset = Sha(images, density_maps, ref_number, transform=train_transforms)

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)



    model = CSRNet(learning_rate=1e-7)


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='train_loss',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        verbose=True
    )

    # create a trainer
    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=100,
        callbacks=[checkpoint_callback],

    )

    # train the model
    trainer.fit(model, train_dataloaders=train_dataloader,val_dataloaders=train_dataloader)



    
if __name__ == '__main__':
    main()
