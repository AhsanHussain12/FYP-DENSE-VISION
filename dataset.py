import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import json

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root *4
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        
        img,target = load_data(img_path,self.train)
        
        #img = 255.0 * F.to_tensor(img)
        
        #img[0,:,:]=img[0,:,:]-92.8207477031
        #img[1,:,:]=img[1,:,:]-95.2757037428
        #img[2,:,:]=img[2,:,:]-104.877445883


        
        
        if self.transform is not None:
            img = self.transform(img)
        return img,target
    
    def collate_fn(self, batch):

        # get the min height and width
        min_height = min([img.shape[1] for img, _ in batch])
        min_width = min([img.shape[2] for img, _ in batch])
        


        # crop the images and the density maps to the min height and width
        img = torch.stack([F.crop(img, 0, 0, min_height, min_width) for img, _ in batch])

        target = np.stack([target[:min_height, :min_width] for _, target in batch])
        
        
        # resize the density maps to 1/8 of the original size
        target = np.stack([cv2.resize(target, (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64 for target in target])
        
        # convert the density maps to torch tensors
        target = torch.from_numpy(target)

        return img, target


# Test the dataset
if __name__ == '__main__':
    with open("part_A_train.json") as f:
        train_data = json.load(f)

    

    train_transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
    
    dataset = listDataset(train_data, train=True, transform=train_transform)
    print(len(dataset))

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True,num_workers=1, collate_fn=dataset.collate_fn)

    for i, (img, target) in enumerate(train_loader):
        print(img.shape)
        print(target.shape)
        break