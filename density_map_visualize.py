import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import h5py

# Define the paths to the image and density map (HDF5 file)
img_path = 'D:/Ashhad/FYP/Dataset/Shanghai/part_A/train_data/images\IMG_100.jpg ' # Replace with your actual image path'
density_map_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')  # Adjust paths accordingly

# Load the image
img = Image.open(img_path)

# Load the density map from the .h5 file
gt_file = h5py.File(density_map_path, 'r')
density_map = np.asarray(gt_file['density'])

# Create a figure with two subplots: one for the image and one for the density map
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[0].axis('off')  # Hide axis

# Display the density map using a colormap for better visualization
ax[1].imshow(density_map, cmap='jet')
ax[1].set_title('Density Map')
ax[1].axis('off')  # Hide axis

# Show the total count in the title
total_count = np.sum(density_map)
ax[1].set_title(f'Density Map (Total Count: {total_count:.2f})')

# Display the figure
plt.show()
