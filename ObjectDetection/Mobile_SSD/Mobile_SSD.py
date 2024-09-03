import torch
import torchvision
import cv2
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np
import importlib.util
import sys
import pandas as pd
import os

# load meerdown
meer_path = 'Data/MeerDown/MeerDown.py'
spec = importlib.util.spec_from_file_location("MeerDown", meer_path)
MeerDown = importlib.util.module_from_spec(spec)
sys.modules["MeerDown"] = MeerDown
spec.loader.exec_module(MeerDown)

class MNDataset(Dataset):
    def __init__(self, image_folder, annotation_file, image_resize = False, image_size=(640, 640)):
        # set class parameters
        self.image_folder = image_folder
        self.annotation_file = annotation_file
        self.image_size = image_size
        self.image_resize = image_resize

        # load annotations
        self.annotations = pd.read_csv(annotation_file)

        # get image files
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # get the image details
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        basename, _ = os.path.splitext(self.image_files[idx])
        vid_name, frame_count_str = basename.split('_frame_')
        
        # normalize and resize image
        image = Image.open(img_name)
        if self.image_resize: 
            image = image.resize(self.image_size)
        image = np.array(image) / 255.0 

        # get corresponding image annotations
        img_annotations = self.annotations[
            (self.annotations['video'] == vid_name) & (self.annotations['frame_number'] == int(frame_count_str))
        ]

        # extract annotations
        boxes = img_annotations[['x1', 'y1', 'x2', 'y2']].values
        labels = np.zeros(len(img_annotations), dtype=int)  # all labels zero for meerkat

        # convert image to tensor
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # convert annotations to dict format expected by torchvision models
        annotations_dict = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor([((x2 - x1) * (y2 - y1)) for x1, y1, x2, y2 in boxes], dtype=torch.float32),
        }

        return image_tensor, annotations_dict

def create_dataloader(image_folder, annotation_file, batch_size=16, shuffle=True, num_workers=4, image_size=(640, 640), image_resize = False):
    dataset = MNDataset(image_folder, annotation_file, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=lambda batch: collate_fn(batch))

def collate_fn(batch):
    images, annotations = zip(*batch)
    return list(images), list(annotations)


class MobileSSD:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', batch_size = 16, image_folder = "Data/MeerDown/raw/frames", annotation_file = "Data/MeerDown/raw/annotations.csv"):
        # load the pretrained model
        self.device = device
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self.model = ssdlite320_mobilenet_v3_large(weights=weights)
        
        # alter the output head to only predict two classes
        in_features = self.model.head.in_channels
        num_classes = 2 
        self.model.head = torchvision.models.detection.ssdlite.SSDLiteHead(in_features, num_classes)
        self.model.to(self.device).eval()

        # create the dataloader
        self.dataloader = create_dataloader(image_folder, annotation_file, batch_size=batch_size)
    
    def train(self, num_epochs=10, lr=0.01):
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            for epoch in range(num_epochs):
                self.model.train()
                running_loss = 0.0
                for images, targets in self.dataloader:
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    optimizer.zero_grad()
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    losses.backward()
                    optimizer.step()

                    running_loss += losses.item()

                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(self.dataloader)}")

    
    def display_video_predictions(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to PIL Image and preprocess
            pil_image = Image.fromarray(frame)
            image_tensor = F.to_tensor(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(image_tensor)[0]
            
            # Get bounding boxes and labels
            boxes = predictions['boxes'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            
            # Draw bounding boxes on the frame
            for box, label, score in zip(boxes, labels, scores):
                if score > 0.5:  # Threshold for confidence
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label}: {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display the frame with bounding boxes
            cv2.imshow('Predictions', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    mn_ssd = MobileSSD()
    mn_ssd.train()