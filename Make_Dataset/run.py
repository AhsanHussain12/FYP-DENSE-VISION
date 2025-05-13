import numpy as np

# Example coordinate array (location)
coordinates = np.array([[29.6225116, 472.92022152],
                        [54.35533603, 454.96602305],
                        [51.79045053, 460.46220626]])

# Example count (number)
count = np.array([[1546]], dtype=np.uint16)

# Create a structured array with fields 'location' and 'number'
image_info = np.array([((coordinates, count))],
                      dtype=[('location', 'O'), ('number', 'O')])

# Display the structured array
print(image_info)
print(image_info["image_info"][0, 0][0, 0][0])


# python visualize.py --model D:\\CSRNET_Repo\\CSRNet\\CSRNet-Light\\t36zoxvu\\checkpoints\\epoch=178-step=214800.ckpt --output_dir D:\CSRNET_Repo\output --image_dir D:\CSRNET_Repo\Shanghai\part_A_final\test_data\images
# python visualize.py --model D:\\CSRNET_Repo\\CSRNet\\CSRNet-Light\\t36zoxvu\\checkpoints\\epoch=178-step=214800.ckpt --video_path D:\\Ashhad\\FYP\\Dataset\\Abnormal-High-density Crowds\\1_Times_Square\\Footage.avi