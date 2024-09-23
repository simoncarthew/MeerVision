import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms, models
import sys
import os

sys.path.append("Data")
from DataManager import DataManager

MD_COCO = os.path.join("Data","MeerDown","raw","annotations.json")
MD_FRAMES = os.path.join("Data","MeerDown","raw","frames")
OBS_COCO = os.path.join("Data","Observed","annotations.json")
OBS_FRAMES = os.path.join("Data","Observed","frames")

class CNNS:
    def __init__(self, model_name, num_classes = 2, model_path = None, pretrained=True, device='cpu'):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.model_name = model_name

        if model_path: # load a previous model
            self.model = self.load_prev_model(model_path)
        else: # load a new model
            self.model = self.load_new_model(model_name,pretrained)

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

        # Move model to device
        model = model.to(self.device)

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
        self.model = self.load_new_model(self.model_name, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        print(f"Model loaded from {file_path}")


    def train(self, train_loader, val_loader, epochs, learning_rate=0.01):
        # set the optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images = torch.stack([image.to(self.device) for image in images])  

                labels = [label['category_id'].to(self.device) for label in labels]  # Ensure category_id is on the right device
                labels = torch.stack(labels)  # Convert list of tensors to one tensor

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
            for images, labels in val_loader:
                images = torch.stack([image.to(self.device) for image in images])  
                labels = [label['category_id'].to(self.device) for label in labels]  # Ensure category_id is on the right device
                labels = torch.stack(labels)  # Convert list of tensors to one tensor
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


if __name__ == "__main__":
    # set data variables
    batch = 32
    num_workers = 8
    data_path = os.path.join("Data","Classification")
    obs_no = 200
    md_z1_trainval_no = 200
    md_z2_trainval_no = 200
    md_test_no = 0
    img_size = (64,64)
    new_cuts = False
    behaviour = False

    # create data_loaders
    data = DataManager(md_coco_path = MD_COCO, md_frames_path = MD_FRAMES, obs_coco_path = OBS_COCO, obs_frames_path = OBS_FRAMES, debug = True)
    train_loader, val_loader, test_loader = data.create_dataloaders(raw_path=data_path, batch = batch, num_workers=num_workers, obs_no=obs_no, md_z1_trainval_no = md_z1_trainval_no, md_z2_trainval_no = md_z2_trainval_no, md_test_no = md_test_no, img_size = img_size, new_cuts = new_cuts, behaviour = behaviour)

    # first test
    resnet = CNNS(model_name="resnet50")
    resnet.train(train_loader=train_loader,val_loader=val_loader,epochs=1)