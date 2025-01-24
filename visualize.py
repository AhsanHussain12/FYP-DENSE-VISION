# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import torch
# from torchvision import transforms
# from model.model import CSRNet
# import Config as cfg
# from argparse import ArgumentParser
# from rich.progress import track
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# import glob

# class ImageDataset(torch.utils.data.Dataset):
#     def __init__(self, image_dir, transform=None):
#         self.image_paths = glob.glob(f"{image_dir}/*.jpg")  # Update the extension if images are different
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         img = Image.open(img_path).convert("RGB")  # Open the image as RGB

#         if self.transform:
#             img = self.transform(img)

#         return img, img_path  # Return the image and its path

# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument('--model', type=str, help='Path to the model')
#     parser.add_argument('--image_dir', type=str, help='Directory with images for visualization')
#     parser.add_argument('--output_dir', type=str, default='outputs/', help='Directory to save visualizations')

#     return parser.parse_args()

# def visualize(args):

#     # ================== Data ==================
#     val_dataset = ImageDataset(image_dir=args.image_dir,
#                                transform=transforms.Compose([
#                                    transforms.ToTensor(), 
#                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                                ]))

#     val_loader = torch.utils.data.DataLoader(
#         val_dataset,
#         batch_size=1,  # Visualize one image at a time
#         shuffle=False
#     )

#     # ================== Model ==================
#     model = CSRNet.load_from_checkpoint(args.model, learning_rate=cfg.learning_rate)
#     model.eval()

#     # ================== Visualization ==================
#     with torch.no_grad():
#         for i, (img, img_path) in track(enumerate(val_loader), total=len(val_loader)):
#             output = model(img)
#             density_map = output.squeeze().cpu().numpy()  # Remove extra dimensions
#             estimated_count = density_map.sum()

#             # Plot the input image and its density map
#             fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#             axs[0].imshow(img.squeeze().permute(1, 2, 0).cpu().numpy())  # Display input image
#             axs[0].set_title("Input Image")
#             axs[1].imshow(density_map, cmap='jet')  # Display density map
#             axs[1].set_title(f"Predicted Density Map\nEstimated Count: {estimated_count:.2f}")
#             plt.show()

#             # Save the visualization if needed
#             output_path = f"{args.output_dir}/density_map_{i}.png"
#             plt.savefig(output_path)
#             print(f"Saved visualization to {output_path}")
#             plt.close(fig)

# if __name__ == '__main__':
#     args = parse_args()
#     visualize(args)



import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torchvision import transforms
from model.model import CSRNet
import Config as cfg
from argparse import ArgumentParser
from rich.progress import track
import cv2  # Import OpenCV
from PIL import Image

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to the model')
    parser.add_argument('--video_path', type=str, help='Path to the input video')
    return parser.parse_args()

def visualize_video(args):
    # ================== Model ==================
    model = CSRNet.load_from_checkpoint(args.model, learning_rate=cfg.learning_rate)
    model.eval()

    # Open the video file
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Transformation for input frames
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ================== Frame Processing ==================
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to PIL image and apply transformations
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = transform(img).unsqueeze(0)  # Add batch dimension

            # Forward pass through the model
            output = model(img)
            density_map = output.squeeze().cpu().numpy()
            estimated_count = density_map.sum()

            # Display the count on the frame
            cv2.putText(frame, f"Estimated Count: {estimated_count:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show the processed frame
            cv2.imshow("Video Frame", frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    visualize_video(args)
