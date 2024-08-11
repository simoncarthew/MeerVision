import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# model type
model_type = "MiDaS_small"

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", model_type)  # or use "MiDaS" for the large model
midas.eval()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


# Load the video file
video_path = "Data/MeerDown/Annotated_videos/22-10-20_C2_06.mp4"  # Replace with your .mp4 file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    print("Processing Frame.")

    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB as OpenCV uses BGR format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to get the depth heatmap
    input_batch = transform(frame_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    
    # Normalize the output
    output_normalized = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # Apply colormap
    output_colored = cv2.applyColorMap((output_normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    
    # Convert the frame to grayscale (black and white)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert grayscale frame to 3-channel BGR format to match depth heatmap format
    frame_bgr_gray = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    
    # Blend the original image and the heatmap with 50% opacity each
    blended_frame = cv2.addWeighted(frame_bgr_gray, 0.4, output_colored, 0.6, 0)
    
    # Display the blended result
    cv2.imshow("Blended Image and Depth Heatmap", blended_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
# out.release()
cv2.destroyAllWindows()
