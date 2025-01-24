import scipy.io

def view_mat_file(mat_file_path):
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_file_path)
    
    # Print the keys in the loaded dictionary
    print("Keys in the .mat file:")
    for key in mat_data.keys():
        print(key)

    # If you want to explore further, print the contents of each key
    print("\nContents of the .mat file:")
    for key in mat_data.keys():
        # Avoid printing the built-in keys of the .mat file
        if not key.startswith('__'):
            print(f"\n{key}:")
            print(mat_data[key])

# Replace with the path to your .mat file
mat_file_path = 'D:\\CSRNET_Repo\\Shanghai\\part_A_final\\train_data\\ground-truth\\GT_IMG_7.mat'  # Update this line
view_mat_file(mat_file_path)
