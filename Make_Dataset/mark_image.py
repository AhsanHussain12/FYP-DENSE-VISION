import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Global variables to store points and image index
points = []
current_image_index = 0

def on_close(event):
    global points, current_image_index, img_paths
    if points:  # Check if any points were recorded
        points_array = np.array(points)  # Convert points to a NumPy array

        # Prepare the structured array with specified data types
        structured_array = np.zeros(1, dtype=[('location', 'O'), ('number', 'O')])
        
        # Directly assign the points_array to the structured array for 'location'
        structured_array['location'][0] = np.array([points_array])  # Shape: (1, N, 2)
        
        # Directly assign the count of points to the structured array for 'number'
        structured_array['number'][0] = np.array([[len(points_array)]], dtype=np.uint16)  # Shape: (1, 1)

        image_filename = img_paths[current_image_index]
        output_filename = image_filename.replace('images', 'ground_truth').replace('.jpg', '.mat')
        
        # Save the structured array in the .mat file
        sio.savemat(output_filename, {'image_info': structured_array}) 
        print(f"Ground truth saved to {output_filename}")


    # Move to the next image
    points = []  # Clear points for the next image
    current_image_index += 1
    plt.close()  # Close the current figure

# Function to capture mouse clicks and record coordinates
def onclick(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        points.append((x, y))
        ax.plot(x, y, 'ro')  # Mark point with red circle
        fig.canvas.draw()

# Load all images from the folder
img_folder = 'D://Ashhad//FYP//Make_Dataset//night_images'
img_paths = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))  # Adjust based on your extension

# Skip images that already have a .mat file
img_paths = [img_path for img_path in img_paths if not os.path.exists(img_path.replace('images', 'ground_truth').replace('.jpg', '.mat'))]

# Loop through each image for annotation
for img_path in img_paths:
    # Load and display the image
    img = plt.imread(img_path)
    
    # Set figure size to a larger size (e.g., 12x8 inches)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    
    # Set the title with the image file name (scene-xxxx)
    image_name = os.path.basename(img_path)
    ax.set_title(image_name)  # Set the title to the image file name

    # Add event listeners for click and close
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('close_event', on_close)  # Connect close event to on_close

    # Show the plot and wait for interaction
    plt.show()

print("Annotation process complete.")
