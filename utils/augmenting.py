import random

import cv2
import numpy as np
from PIL import Image


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
        

    