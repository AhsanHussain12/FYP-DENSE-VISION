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
import json
import dataset
import Config as cfg
import image
from argparse import ArgumentParser

import pytorch_lightning as pl

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to the model')

    return parser.parse_args()

def test(args):

    # ================== Data ==================
    with open(cfg.val_json) as f:
        val_list = json.load(f)

    val_dataset = dataset.listDataset(val_list,
                                        shuffle=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225]),
                                        ]),
                                        train=False,
                                        batch_size=cfg.batch_size,
                                        num_workers=cfg.num_workers)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True)
    
    # ================== Model ==================
    model = CSRNet.load_from_checkpoint(args.model, learning_rate=cfg.learning_rate)
    model.eval()

    # ================== Test ==================
    mae = 0

    with torch.no_grad():
        for i, (img, target) in tqdm(enumerate(val_loader), colour='green', desc='Testing', total=len(val_loader)):
            output = model(img)
            mae += abs(output.sum().item() - target.sum().item())
    
    # Test results
    print(f"----- Test results -----")
    print(f"MAE: {mae / len(val_dataset)}")




if __name__ == '__main__':
    args = parse_args()
    test(args)