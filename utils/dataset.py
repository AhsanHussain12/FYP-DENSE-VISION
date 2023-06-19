import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import cv2
import numpy as np
from PIL import Image
import h5py

# Your original functions
def numpy_crop(np_array, i, j, h, w):
    return np_array[i:i+h, j:j+w]

def augment_images(images, density_maps, ref_number):
    augmented_images = []
    augmented_density_maps = []
    num_patches_per_image = ref_number // len(images)
    for image, density_map in zip(images, density_maps):
        for _ in range(num_patches_per_image):
            image = np.asarray(image)
            crop_size = (image.shape[0] // 4, image.shape[1] // 4)
            rand_i,rand_j,rand_h,rand_w = random.randint(0, image.shape[0] - crop_size[0]), random.randint(0, image.shape[1] - crop_size[1]), crop_size[0], crop_size[1]
            augmented_image = numpy_crop(image, rand_i, rand_j, rand_h, rand_w)
            augmented_density_map = numpy_crop(density_map, rand_i, rand_j, rand_h, rand_w)
            assert augmented_image.shape[:2] == augmented_density_map.shape[:2]
            # permute the channels to match the input format of the pretrained model
            augmented_image = augmented_image.transpose(2, 0, 1)

            augmented_images.append(augmented_image)
            augmented_density_maps.append(augmented_density_map)     
    return augmented_images, augmented_density_maps

# Dataset class
class Sha(Dataset):
    def __init__(self, images, density_maps, ref_number,transform=None):
        self.images, self.density_maps = augment_images(images, density_maps, ref_number)
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx].copy()).float()
        density_map = torch.from_numpy(self.density_maps[idx].copy()).float().unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, density_map

# Custom collate_fn to match sizes
class MatchSizeTransform:
    def __call__(self, image1, image2):
        min_height = min(image1.shape[1], image2.shape[1])
        min_width = min(image1.shape[2], image2.shape[2])

        print(f"Min height: {min_height}")
        print(f"Min width: {min_width}")

        crop_transform = transforms.CenterCrop((min_height, min_width))
        image1 = crop_transform(image1)
        image2 = crop_transform(image2)
        return image1, image2

def collate_fn(batch):
    images, density_maps = zip(*batch)

    min_height = min([img.shape[1] for img in images])
    min_width = min([img.shape[2] for img in images])

    # Create a transform that crops to these dimensions
    crop_transform = transforms.CenterCrop((min_height, min_width))

    cropped_images = [crop_transform(img) for img in images]
    cropped_density_maps = [crop_transform(dm) for dm in density_maps]

    # resize the density maps to 1/8 the size of the original image. use torchvision.transforms.Resize and multiply by 64
    resize_transform = transforms.Resize((min_height//8, min_width//8))
    cropped_density_maps = [resize_transform(dm) * 64 for dm in cropped_density_maps]

    return torch.stack(cropped_images), torch.stack(cropped_density_maps)





