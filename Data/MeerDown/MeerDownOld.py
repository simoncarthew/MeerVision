import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import shutil
import yaml
import matplotlib.pyplot as plt
from multiprocessing import Pool
import cv2

class MeerDown(Dataset):
    def __init__(self, image_folder = "Data/MeerDown/raw/frames", annotation_file = "Data/MeerDown/raw/annotations.csv", image_size=(640, 640)):
        self.image_folder = image_folder
        self.annotation_file = annotation_file
        self.image_size = image_size
        
        # load annotations
        self.annotations = pd.read_csv(annotation_file)
        
        # list of image files
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        basename, _ = os.path.splitext(self.image_files[idx])
        vid_name, frame_count_str = basename.split('_frame_')
        image = Image.open(img_name).resize(self.image_size)
        image = np.array(image) / 255.0 

        img_annotations = self.annotations[
            (self.annotations['video'] == vid_name) & (self.annotations['frame_number'] == frame_count_str)
        ]

        # Extract additional annotations
        object_ids = img_annotations['object_ID'].values
        behaviour_indices = img_annotations['behaviour_index'].values
        occluded = img_annotations['occluded'].values
        boxes = img_annotations[['x1', 'y1', 'x2', 'y2']].values

        # Combine all annotations into a single array
        annotations = np.column_stack([boxes, object_ids, behaviour_indices, occluded])
        
        return image, annotations
    
    def create_dataloader(self, batch_size=16, shuffle=True, num_workers=4):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        images, annotations = zip(*batch)
        images = torch.stack(images, dim=0)
        annotations = [torch.tensor(a, dtype=torch.float32) for a in annotations]
        return images, annotations

    def create_yolo_dataset(self, yolo_dir='Data/MeerDown/yolo', train_set = "C2", total_val_in_train = 14140/2, total_val_images = 14140/2):
        # check if data exists and if they want to redo data
        redo = False
        if os.path.exists(yolo_dir):
            if input("Yolo dataset already exists, would you like to delete (y/n)? ") == "y":
                shutil.rmtree(yolo_dir)
                redo = True
        else:
            redo = True

        if redo:
            # no val images
            num_val_images = 0
            num_val_in_train = 0

            # make yolo directory
            os.makedirs(yolo_dir, exist_ok=True)

            # make subdirectories
            yolo_images_dir = os.path.join(yolo_dir, 'images')
            yolo_labels_dir = os.path.join(yolo_dir, 'labels')
            os.makedirs(os.path.join(yolo_images_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(yolo_images_dir, "val"), exist_ok=True)
            os.makedirs(os.path.join(yolo_labels_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(yolo_labels_dir, "val"), exist_ok=True)

            # Create the .yaml file
            yaml_content = {
                'path': os.path.abspath(yolo_dir),
                'train': 'images/train',
                'val': 'images/val',
                'test': '',
                'names': {0: 'meerkat'}
            }
            with open(os.path.join(yolo_dir, 'dataset.yaml'), 'w') as yaml_file:
                yaml.dump(yaml_content, yaml_file, default_flow_style=False)
            
            # iterate over every image
            for idx in range(len(self)):
                # get vid_name and frame_count
                img_name = self.image_files[idx]
                basename, _ = os.path.splitext(img_name)
                vid_name, frame_count_str = basename.split('_frame_')

                # filter for relevant annotations
                img_annotations = self.annotations[
                    (self.annotations['video'] == vid_name) & (self.annotations['frame_number'] == int(frame_count_str))
                ]

                 # open image and get width 
                img = Image.open(os.path.join(self.image_folder, img_name))
                img_width, img_height = img.size
                
                # create yolo annotations
                yolo_annotations = []
                for _, row in img_annotations.iterrows():
                    # extract bounding box coordinates and normalize them
                    x_center = (row['x1'] + row['x2']) / 2 / img_width
                    y_center = (row['y1'] + row['y2']) / 2 / img_height
                    width = (row['x2'] - row['x1']) / img_width
                    height = (row['y2'] - row['y1']) / img_height
                    
                    # yolo_annotation = f"{row['behaviour_index']} {x_center} {y_center} {width} {height}"
                    yolo_annotation = f"{0} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    yolo_annotations.append(yolo_annotation)

                # check if train or val
                if train_set in vid_name : train_val = "train" 
                elif num_val_in_train < total_val_in_train: 
                    train_val = "train"
                    num_val_in_train += 1
                else: train_val = "val"

                if (num_val_images < total_val_images and train_val == "val") or train_val == "train":

                    # write the annotations to a .txt file
                    txt_filename = os.path.join(yolo_labels_dir, train_val, f"{basename}.txt")
                    with open(txt_filename, 'w') as f:
                        f.write("\n".join(yolo_annotations))
                    
                    # Save the image to the YOLO directory
                    img = img.resize(self.image_size)
                    img.save(os.path.join(yolo_images_dir, train_val, img_name))

                    if idx % 500 == 0:
                        print("Completed " + str(idx) + "/" + str(len(self)) + " images.")

                    if train_val == "val":
                        num_val_images += 1

    def display_yolo(self,yolo_dir='Data/MeerDown/yolo_reduced_val', image_size=(640, 640), fps=10):
        # Load YOLO annotations and images
        images_dir = os.path.join(yolo_dir, 'images', 'train')
        labels_dir = os.path.join(yolo_dir, 'labels', 'train')

        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')])
        
        for img_name in image_files:
            # Load the image
            img_path = os.path.join(images_dir, img_name)
            img = cv2.imread(img_path)

            # Load the corresponding label file
            basename, _ = os.path.splitext(img_name)
            label_path = os.path.join(labels_dir, f"{basename}.txt")
            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                annotations = f.readlines()

            # Plot each bounding box on the image
            for annotation in annotations:
                class_id, x_center, y_center, width, height = map(float, annotation.split())
                x_center, y_center, width, height = int(x_center * image_size[0]), int(y_center * image_size[1]), int(width * image_size[0]), int(height * image_size[1])
                x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
                x2, y2 = int(x_center + width / 2), int(y_center + height / 2)

                # Draw the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'Class {int(class_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the image
            cv2.imshow('Labeled Image', img)
            
            # Wait for the duration of one frame (in milliseconds)
            time_per_frame = int(1000 / fps)
            cv2.waitKey(time_per_frame)

        # Release the display window
        cv2.destroyAllWindows()

if __name__ == "__main__":
    md = MeerDown()
    # md.create_yolo_dataset(yolo_dir = 'Data/MeerDown/yolo/yolo_val_mix_half')
    # md.display_yolo()
    data = md.create_dataloader()