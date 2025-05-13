import os

def rename_images(folder_path):
    # Get a list of all files in the specified folder
    files = sorted(os.listdir(folder_path))

    # Loop through each file and rename it, starting from 1801
    for i, filename in enumerate(files, start=1882):
        # Set the new file name with zero-padded numbers
        new_name = f"scene-{i:05d}.mat"  # Adjust extension if necessary

        # Create the full path for the current and new file names
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed '{filename}' to '{new_name}'")

# Specify the folder path where your images are stored
folder_path = r'D:\ground_truth'  # Replace with your actual folder path
rename_images(folder_path)
