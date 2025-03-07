import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import cv2
import numpy as np
from torchvision import transforms
from model.model import CSRNet
import Config as cfg
from argparse import ArgumentParser
from PIL import Image

class VideoVisualizer:
    def __init__(self, model_path, video_path, device=None):
        # Select device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        print(f"Using device: {self.device}")

        # Load the model
        self.model = CSRNet.load_from_checkpoint(model_path, learning_rate=cfg.learning_rate).to(self.device)
        self.model.eval()

        # Open video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video.")

        # Image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def visualize(self):
        frame_count = 0
        total_time = 0

        with torch.no_grad():
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Convert frame to PIL image and apply transformations
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = self.transform(img).unsqueeze(0).to(self.device)  # Add batch dimension   

                # Measure inference time
                start_time = time.time()
                output = self.model(img)
                end_time = time.time()

                # Compute FPS
                inference_time = end_time - start_time
                fps = 1 / inference_time if inference_time > 0 else 0
                total_time += inference_time
                frame_count += 1

                # Process model output
                density_map = output.squeeze().cpu().numpy()
                estimated_count = density_map.sum()

                # Normalize density map for visualization
                density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)  # Avoid div by zero
                density_map = (density_map * 255).astype(np.uint8)  # Scale to 0-255
                density_map = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)  # Apply colormap

                # Resize density map to match video frame size
                density_map = cv2.resize(density_map, (frame.shape[1], frame.shape[0]))

                # Display estimated count & FPS on frame
                cv2.putText(frame, f"Estimated Count: {estimated_count:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Concatenate frame & density map side by side
                combined_display = np.hstack((frame, density_map))

                # Show combined output
                cv2.imshow("Video + Density Map", combined_display)

                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Calculate and display average FPS
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"Average FPS: {avg_fps:.2f}")

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    visualizer = VideoVisualizer(args.model, args.video_path)
    visualizer.visualize()
