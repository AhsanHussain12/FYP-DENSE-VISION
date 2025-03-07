import cv2
import numpy as np
import threading
import time
import torch
import argparse
import tkinter as tk
import Config as cfg
from tkinter import Label, Frame
from torchvision import transforms
from PIL import Image, ImageTk
from model.model import CSRNet  # Ensure CSRNet is properly imported

# ===================== Argument Parsing =====================
parser = argparse.ArgumentParser(description="Real-time Crowd Counting with CSRNet")
parser.add_argument("--model", type=str, required=True, help="Path to the trained CSRNet model checkpoint (.pth)")
parser.add_argument("--video", type=str, required=True, help="Path to the input video file")
args = parser.parse_args()

# ===================== Load CSRNet Model =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSRNet.load_from_checkpoint(args.model, learning_rate=cfg.learning_rate).to(device)
model.eval()

# Open Video
cap = cv2.VideoCapture(args.video)
global_frame = None
crowd_count = 0
frames_processing_interval=0.25
# Tkinter GUI Setup
root = tk.Tk()
root.title("Crowd Counting System")

# ===================== Layout Configuration (Fixed for Side-by-Side) =====================
main_frame = Frame(root)
main_frame.grid(row=0, column=0, padx=10, pady=10)

video_label = Label(main_frame)
video_label.grid(row=0, column=0, padx=10, pady=10)  # Video on the left

density_label = Label(main_frame)
density_label.grid(row=0, column=1, padx=10, pady=10)  # Density map on the right

count_label = Label(root, text="Crowd Count: 0", font=("Arial", 24), fg="green")
count_label.grid(row=1, column=0, columnspan=2, pady=10)


# Define Transformations for Model Input
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ===================== Video Display =====================
def update_video():
    global global_frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        global_frame = frame.copy()
        frame_resized = cv2.resize(frame, (1000, 700))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img)
        video_label.config(image=img_tk)
        video_label.image = img_tk
        
        time.sleep(0.03)  # Slight delay to match frame rate roughly 30 fps

# ===================== Frame Processing for Crowd Counting =====================
def process_frames():
    global crowd_count, global_frame
    while True:
        if global_frame is not None:
            frame = global_frame.copy()
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img)
                density_map = output.squeeze().cpu().numpy()
                crowd_count = int(density_map.sum())
                print(f"Crowd Count: {crowd_count}")
            
            # density_map_resized = cv2.resize(density_map, (1000, 700))
            density_map_resized = cv2.resize(density_map, (1000, 700), interpolation=cv2.INTER_CUBIC)
            density_map_resized = (density_map_resized / density_map_resized.max() * 255).astype(np.uint8)
            density_map_resized = cv2.applyColorMap(density_map_resized, cv2.COLORMAP_JET)
            
            img_density = Image.fromarray(cv2.cvtColor(density_map_resized, cv2.COLOR_BGR2RGB))
            img_tk_density = ImageTk.PhotoImage(img_density)
            density_label.config(image=img_tk_density)
            density_label.image = img_tk_density  # Ensure reference is maintained
            
            count_label.config(text=f"Crowd Count accurate to {frames_processing_interval} Second: {crowd_count}")
        time.sleep(frames_processing_interval)

# ===================== Start Threads =====================
if __name__ == "__main__":
    video_thread = threading.Thread(target=update_video, daemon=True)
    count_thread = threading.Thread(target=process_frames, daemon=True)
    
    video_thread.start()
    count_thread.start()
    
    root.mainloop()
    cap.release()
