import cv2
import numpy as np
import threading
import time
# import torch
import tkinter as tk
from tkinter import Label, Frame
# from torchvision import transforms
from PIL import Image, ImageTk

# Load CSRNet Model (Assuming model is already trained and loaded)
# model = torch.load("csrnet_model.pth", map_location=torch.device("cpu"))
# model.eval()  # Set model to evaluation mode

cap = cv2.VideoCapture("C:\\Users\\DELL\\Desktop\\FYP\\video.mp4")
global_frame = None  # Shared variable for latest frame
crowd_count = 0  # Crowd count storage

root = tk.Tk()
root.title("Crowd Counting System")

# ===================== Layout Configuration =====================
main_frame = Frame(root)
main_frame.grid(row=0, column=0, padx=10, pady=10)

video_frame = Frame(main_frame)
video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

count_frame = Frame(main_frame)
count_frame.grid(row=0, column=1, padx=10, pady=10, sticky="e")

video_label = Label(video_frame)
video_label.pack()

count_label = Label(count_frame, text="Crowd Count: 0", font=("Arial", 24), fg="green")
count_label.pack()


# ===================== Video Display =====================
def update_video():
    global global_frame
    ret, frame = cap.read()
    if ret:
        global_frame = frame.copy()  # Store latest frame
        frame = cv2.resize(frame, (1000, 700))  # Resize to square
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(img)
        video_label.config(image=img)
        video_label.image = img

    root.after(30, update_video)

# ===================== Frame Processing for Crowd Counting =====================
def process_frames():
    global crowd_count, global_frame

    while True:
        frames = []
        
        for _ in range(3):  # Capture 3 frames
            if global_frame is not None:
                frames.append(global_frame.copy())  # Use latest available frame
            time.sleep(0.5)

        if frames:
            crowd_count = np.random.randint(10, 100)  # Stub crowd count
            count_label.config(text=f"Crowd Count: {crowd_count}")

        time.sleep(2)

# ===================== Start Threads =====================
if __name__ == "__main__":
    video_thread = threading.Thread(target=update_video, daemon=True)
    count_thread = threading.Thread(target=process_frames, daemon=True)

    video_thread.start()
    count_thread.start()

    root.mainloop()
    cap.release()
