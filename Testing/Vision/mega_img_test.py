import cv2
import torch
import os
from PytorchWildlife.models import detection as pw_detection

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Load MegaDetector v5
detection_model = pw_detection.MegaDetectorV5(device=device)

# Input and output directories
input_folder = "/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/data/open_meerkat_images/val"
output_folder = "/home/simon/OneDrive/University/Fourth_Year/Second Semester/EEE4022S/MeerVision/Testing/Vision/results/mega_img_det"
os.makedirs(output_folder, exist_ok=True)

# Process each .jpg file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Read the image
        image = cv2.imread(input_path)
        height, width, _ = image.shape

        # Resize the image for the model
        resized_image = cv2.resize(image, (1280, 1280))
        img_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            detection_result = detection_model.single_image_detection(img_tensor[0])

        detections = detection_result['detections']
        print(f"Detections for {filename}: {detections}")

        for i in range(len(detections.confidence)):
            category = detections.class_id[i]  # Class ID
            bbox = detections.xyxy[i]  # Bounding box
            conf = detections.confidence[i]  # Confidence score

            # Scale back to original size
            x1, y1, x2, y2 = [int(coord * width / 1280) if j % 2 == 0 else int(coord * height / 1280) for j, coord in enumerate(bbox)]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Animal: {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the processed image
        cv2.imwrite(output_path, image)
        print(f"Processed image saved as {output_path}")

print("Image processing complete.")
