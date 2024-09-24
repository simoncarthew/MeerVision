import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms, models
import sys
import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("Data")
from DataManager import DataManager

MD_COCO = os.path.join("Data","MeerDown","raw","annotations.json")
MD_FRAMES = os.path.join("Data","MeerDown","raw","frames")
OBS_COCO = os.path.join("Data","Observed","annotations.json")
OBS_FRAMES = os.path.join("Data","Observed","frames")

class CNNS:
    def __init__(self, model_name, num_classes = 2, model_path = None, pretrained=True, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.model_name = model_name

        if model_path: # load a previous model
            self.model = self.load_prev_model(model_path)
        else: # load a new model
            self.model = self.load_new_model(model_name,pretrained)

        self.model.to(self.device)

    def load_new_model(self, model_name, pretrained):
        weights = None
        if pretrained:
            if model_name == 'resnet50':
                weights = models.ResNet50_Weights.DEFAULT
            elif model_name == 'vgg16':
                weights = models.VGG16_Weights.DEFAULT
            elif model_name == 'shufflenet_v2':
                weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT
            elif model_name == 'mobilenet_v2':
                weights = models.MobileNet_V2_Weights.DEFAULT

        # Load the selected architecture
        if model_name == 'resnet50':
            model = models.resnet50(weights=weights)
        elif model_name == 'efficientnet_b0':
            model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=pretrained)
        elif model_name == 'vgg16':
            model = models.vgg16(weights=weights)
        elif model_name == 'shufflenet_v2':
            model = models.shufflenet_v2_x1_0(weights=weights)
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=weights)
        else:
            raise ValueError(f"Model {model_name} not supported")

        # Adjust final layer for specific class
        if isinstance(model, torch.nn.modules.container.Sequential):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.num_classes)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Linear):
                model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)
            else:
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.num_classes)
        elif hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        else:
            raise ValueError(f"Unexpected model architecture for {model_name}")

        return model


    def load_prev_model(self, file_path):
        # get previous model information
        checkpoint = torch.load(file_path, map_location=self.device)
        self.model_name = checkpoint['model_name']
        self.num_classes = checkpoint['num_classes']
        
        # load previous models dict to correct architecture
        model = self.load_new_model(self.model_name, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = self.model.to(self.device)
        
        return model


    def train(self, train_loader, val_loader, epochs, learning_rate=0.01):
        # set the optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in tqdm(train_loader):
                labels = labels[0]['category_id'].to(self.device)
                images = torch.stack([image.to(self.device) for image in images])  

                optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracy = 100 * correct / total
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

            self.validate(val_loader)

    def validate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                labels = labels[0]['category_id'].to(self.device)
                images = torch.stack([image.to(self.device) for image in images]) 
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f'Validation Accuracy: {val_accuracy:.2f}%')

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for images in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        return predictions

    def save_model(self, file_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes
        }, file_path)
        print(f"Model saved to {file_path}")

    def unnormalize(self,image, mean, std):
        """Undo the normalization for plotting."""
        mean = np.array(mean).reshape(1, 1, 3)
        std = np.array(std).reshape(1, 1, 3)
        image = (image * std) + mean
        return np.clip(image, 0, 1)

    def show_predictions(self, dataloader, num_images=5):
        self.model.eval()
        images, labels = next(iter(dataloader))
        images = images.to(self.device)
        labels = labels[0]['category_id'].to(self.device)  # assuming labels are in the first element

        # Randomly select images and their corresponding labels
        indices = random.sample(range(len(images)), num_images)
        selected_images = images[indices]
        selected_labels = labels[indices].cpu().numpy()

        with torch.no_grad():
            outputs = self.model(selected_images)
            _, predicted = torch.max(outputs, 1)

        # Convert images to numpy arrays for plotting
        selected_images = selected_images.cpu().numpy()  # Shape: (N, C, H, W)
        selected_images = selected_images.transpose((0, 2, 3, 1))  # Change to (N, H, W, C)

        # Unnormalize the images
        mean = [0.485, 0.456, 0.406]  # ImageNet mean
        std = [0.229, 0.224, 0.225]   # ImageNet std
        selected_images = [self.unnormalize(img, mean, std) for img in selected_images]

        # Plotting
        plt.figure(figsize=(15, 10))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(selected_images[i])  # Correctly formatted for plt.imshow
            plt.axis('off')
            plt.title(f"Pred: {predicted[i].item()}, Actual: {selected_labels[i]}")
        plt.savefig('predictions.png')  # Save the figure to a file instead of displaying it
        plt.close()  # Close the plot to free memory


if __name__ == "__main__":
    # set data variables
    batch = 16
    num_workers = 1
    data_path = os.path.join("Data","Classification")
    obs_no = 100
    md_z1_trainval_no = 100
    md_z2_trainval_no = 100
    md_test_no = 0
    img_size = (64,64)
    new_cuts = True
    behaviour = False
    snap_no=300
    snap_test_no = 50

    # create data_loaders
    # dm = DataManager(md_coco_path="Data/MeerDown/raw/behaviour_annotations.json")
    dm = DataManager()
    train_loader, val_loader, test_loader = dm.create_dataloaders(batch=batch,num_workers=num_workers,obs_no = obs_no, md_z1_trainval_no=md_z1_trainval_no,md_z2_trainval_no=md_z2_trainval_no, snap_no=snap_no,snap_test_no=snap_test_no,new_cuts=new_cuts,behaviour=behaviour)

    # first test
    resnet = CNNS(model_name="resnet50",num_classes=2)
    resnet.train(train_loader=train_loader,val_loader=val_loader,epochs=2)
    resnet.show_predictions(test_loader,5)