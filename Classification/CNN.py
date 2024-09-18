import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join('Data'))

from DataManager import DataManager

class CNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, layers=None):
        super(CNN, self).__init__()
        
        self.num_classes = num_classes
        self.layers = layers if layers is not None else self.default_layers(input_channels, num_classes)
        self.model = nn.Sequential(*self.layers)
        
    def default_layers(self, input_channels, num_classes):
        return [
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        ]
    
    def forward(self, x):
        return self.model(x)

class Trainer:
    def __init__(self, model=None, input_channels=3, num_classes=1, learning_rate=0.01, optimizer_type='SGD'):
        # Instantiate a new model or load an existing one
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model if model else CNN(input_channels, num_classes).to(self.device)
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.optimizer = self._set_optimizer()
        self.criterion = nn.CrossEntropyLoss()
    
    def _set_optimizer(self):
        if self.optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer type")
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print("Model loaded from", path)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved to", path)
    
    def train(self, train_loader, val_loader=None, epochs=10):
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Track the loss and accuracy
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/total:.4f}, Accuracy: {100 * correct / total:.2f}%")
            
            if val_loader:
                self.validate(val_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')
        self.model.train()

    def test(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Test Accuracy: {100 * correct / total:.2f}%')
    
    def predict(self, image):
        self.model.eval()
        image = image.unsqueeze(0).to(self.device) 
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()
    
    def set_layers(self, layers):
        self.model = nn.Sequential(*layers).to(self.device)
    
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = self._set_optimizer()
    
    def set_optimizer(self, optimizer_type):
        self.optimizer_type = optimizer_type
        self.optimizer = self._set_optimizer()
    
    def set_num_classes(self, num_classes):
        self.model.num_classes = num_classes
        self.model = CNN(num_classes=num_classes).to(self.device)

# Example usage (assuming DataLoader objects `train_loader`, `val_loader`, and `test_loader` exist)
# trainer = CNNTrainer(num_classes=10, learning_rate=0.001, optimizer_type='adam')
# trainer.train(train_loader, val_loader=val_loader, epochs=10)
# trainer.test(test_loader)