import torch
import torchvision
import cv2
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDSmall
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np

class MobileNetV2SSD:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # load the pretrained model
        self.device = device
        self.model = ssdlite320_mobilenet_v3_large(pretrained=True)
        self.model.to(self.device).eval()
        
    def load_dataset(self, dataset, batch_size=4, shuffle=True):
        """
        Load a dataset for training.
        
        Parameters:
        - dataset: Custom dataset instance that should return images and labels
        - batch_size: Number of samples per batch
        - shuffle: Whether to shuffle the dataset
        
        Returns:
        - DataLoader instance
        """
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
    def train(self, dataloader, num_epochs=10, lr=0.001):
        """
        Train the model.
        
        Parameters:
        - dataloader: DataLoader instance providing the training data
        - num_epochs: Number of epochs for training
        - lr: Learning rate
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for images, targets in dataloader:
                images = [F.to_tensor(img).to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                
                running_loss += losses.item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")
    
    def display_video_predictions(self, video_path):
        """
        Load a video and display predicted bounding boxes.
        
        Parameters:
        - video_path: Path to the video file
        """
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
