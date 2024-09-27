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
from PIL import Image
import json
import copy
import glob
import time
from sklearn.metrics import confusion_matrix

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
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=True)
        self.model_name = checkpoint['model_name']
        self.num_classes = checkpoint['num_classes']
        
        # load previous models dict to correct architecture
        model = self.load_new_model(self.model_name, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model


    def train(self, train_loader, val_loader, epochs, learning_rate=0.01, test_loader = None, inference_path = None):
        # initialize results variables
        results = {"train_acc":[],"train_loss":[],"val_acc":[],"val_loss":[]}

        # set the optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # best model info
        best_val_accuracy = 0.0
        best_model_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
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

            # calculate accuracy and loss
            train_accuracy = 100 * correct / total
            training_loss = running_loss/len(train_loader)

            # validate the model
            val_loss, val_accuracy, _ = self.validate(val_loader)

            # save
            results["train_acc"].append(train_accuracy)
            results["train_loss"].append(training_loss)
            results["val_acc"].append(val_accuracy)
            results["val_loss"].append(val_loss)

            # Check if the current model is the best based on validation accuracy
            if val_accuracy >= best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = copy.deepcopy(self.model.state_dict())  # Save the best model's state
                results["best_epoch"] = epoch

            # print results
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {training_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            torch.cuda.empty_cache()

        # Set self.model to the best model
        self.model.load_state_dict(best_model_state)

        # test the model
        if test_loader:
            _ , test_acc, conf_matrix = self.validate(test_loader)
            results["test_acc"] = test_acc
            results["conf_matrix"] = conf_matrix.tolist()

        # get the inference times
        if inference_path:
            results["inference"] = self.inference_test(inference_path)

        return results

    def validate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss()  # Define the loss function

        # Store true and predicted labels for confusion matrix
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels[0]['category_id'].to(self.device)
                images = torch.stack([image.to(self.device) for image in images])

                # Forward pass to get outputs
                outputs = self.model(images)
                
                # Calculate the loss
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)  # Accumulate loss

                # Get predictions
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Store the true and predicted labels for confusion matrix
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Calculate validation metrics
        val_loss = running_loss / total  # Calculate average loss
        val_accuracy = 100 * correct / total  # Calculate accuracy

        # Create confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        return val_loss, val_accuracy, conf_matrix

    def predict(self, image_path, img_size=64):
        # Define the transforms used during training
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

        # Load the image and apply the transformations
        image = Image.open(image_path).convert('RGB') 
        image = transform(image).unsqueeze(0) 

        # Move the image to the device (GPU/CPU)
        image = image.to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Get the model's prediction
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)

        # Return the predicted class
        return predicted.item()

    def inference_test(self, img_dir):
        image_files = glob.glob(os.path.join(img_dir, '*.jpg'))
        total_time = 0
        num_images = len(image_files)

        for image_path in image_files:
            start_time = time.time()  # Start the timer
            predicted_class = self.predict(image_path)  # Predict the class
            end_time = time.time()  # End the timer
            
            inference_time = end_time - start_time
            total_time += inference_time

        if num_images > 0:
            average_inference_time = total_time / num_images
        else:
            print("No images found in the specified directory.")

        return average_inference_time

    def save_model(self, file_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes
        }, file_path)
        print(f"Model saved to {file_path}")

    def plot_predictions(self, directory_path, coco_name = 'test_coco.json', num_images=16, img_size=64):
        # Load the COCO annotations file
        coco_json_path = os.path.join(directory_path, coco_name)
        
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Get image file names and their actual class labels
        images_info = coco_data['images']
        annotations_info = coco_data['annotations']
        categories_info = coco_data['categories']
        
        # Create a dictionary mapping image IDs to actual category IDs
        image_id_to_category = {ann['image_id']: ann['category_id'] for ann in annotations_info}
        
        # Create a dictionary mapping category IDs to category names
        category_id_to_name = {cat['id']: cat['name'] for cat in categories_info}
        
        # Randomly select images to process
        selected_images = random.sample(images_info, num_images)
        
        # Store the images, actual classes, and predicted classes
        actual_classes = []
        predicted_classes = []
        selected_images_paths = []
        
        # Loop through selected images and predict classes
        for img_info in selected_images:
            img_id = img_info['id']
            img_filename = img_info['file_name']
            img_path = os.path.join(directory_path, img_filename)
            
            # Get the actual class label
            actual_class_id = image_id_to_category[img_id]
            actual_class_name = category_id_to_name[actual_class_id]
            actual_classes.append(actual_class_name)
            
            # Predict the class using the previously defined `predict` function
            predicted_class_id = self.predict(img_path, img_size=img_size)
            predicted_class_name = category_id_to_name[predicted_class_id]
            predicted_classes.append(predicted_class_name)
            
            # Store the image path for plotting
            selected_images_paths.append(img_path)
        
        # Plot the images with their predicted and actual labels
        plt.figure(figsize=(20, 5))
        for i, img_path in enumerate(selected_images_paths):
            img = Image.open(img_path)
            plt.subplot(2, 8, i + 1)
            plt.imshow(img)
            plt.title(f"Pred: {predicted_classes[i]}\nActual: {actual_classes[i]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('predictions.jpg')


if __name__ == "__main__":
    # set data variables
    batch = 4
    num_workers = 1
    data_path = os.path.join("Data","Classification")
    obs_no = 100
    obs_test_no = 10
    md_z1_trainval_no = 50
    md_z2_trainval_no = 50
    md_test_no = 0
    img_size = (64,64)
    behaviour = False
    snap_no=250
    snap_test_no = 10
    epochs = 20

    # create data_loaders
    dm = DataManager(perc_val = 0.2)
    train_loader, val_loader, test_loader = dm.create_dataloaders(batch=batch,num_workers=num_workers,obs_no = obs_no, obs_test_no=obs_test_no, md_z1_trainval_no=md_z1_trainval_no,md_z2_trainval_no=md_z2_trainval_no, snap_no=snap_no,snap_test_no=snap_test_no,behaviour=behaviour,img_size=img_size)
    
    # md_coco_path="Data/MeerDown/raw/behaviour_annotations.json"
    # obs_coco_path="Data/Observed/behaviour_annotations.json"
    # dm = DataManager(md_coco_path=md_coco_path,obs_coco_path=obs_coco_path)
    # train_loader, val_loader, test_loader = dm.create_dataloaders(batch=batch,num_workers=num_workers,obs_no = obs_no, obs_test_no=obs_test_no, md_z1_trainval_no=md_z1_trainval_no,md_z2_trainval_no=md_z2_trainval_no, snap_no=snap_no,snap_test_no=snap_test_no,behaviour=behaviour,img_size=img_size)

    model = CNNS(model_name="resnet50",num_classes=2,pretrained=True)
    results = model.train(train_loader=train_loader,val_loader=val_loader,epochs=epochs,test_loader=test_loader, inference_path="Data/Classification/Binary/cut_images/test")
    model.save_model("ObjectDetection/Megadetector/resnet_test.pth")
    # model.plot_predictions("Data/Classification/Binary/cut_images/test")