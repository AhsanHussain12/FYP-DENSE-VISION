import scipy.io as sio

# Load the .mat file
data = sio.loadmat('C:/Users/binary computers/Desktop/Make_Dataset/ground_truth/scene-00003.mat')

# Print the keys in the loaded data
print("Keys in the .mat file:")
for key in data.keys():
    print(key)

# Access and print the specific content based on the keys
# Replace 'image_info' with the actual key you want to examine
if 'image_info' in data:
    image_info = data['image_info']
    print("\nContents of 'image_info':")
    print(image_info[0,0][0][0])
else:
    print("\nKey 'image_info' not found in the .mat file.")
