import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt

class MeerDown:
    def __init__(self, image_folder = "DataMeerDown/frames", annotation_file = "DataMeerDown/annotations", image_size=(256, 256)):
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
        image = np.array(image) / 255.0  # Normalize to [0, 1]

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

    def to_tensorflow_dataset(self, batch_size=32):
        def gen():
            for i in range(len(self)):
                image, annotations = self[i]
                yield image, annotations
        
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(self.image_size[0], self.image_size[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 7), dtype=tf.float32) 
            )
        )
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset


# Example usage
image_folder = 'Data/MeerDown/frames'
annotation_file = 'Data/MeerDown/annotations.csv'

dataset = MeerDown(image_folder, annotation_file)

# TensorFlow DataLoader
tf_dataset = dataset.to_tensorflow_dataset(batch_size=32)

# Iterate over the dataset
for images, boxes in tf_dataset:
    print("Batch of images shape:", images.shape)
    print("Batch of bounding boxes shape:", boxes.shape)

    # Display the first image in the batch
    for i in range(images.shape[0]):
        plt.figure(figsize=(8, 8))
        plt.imshow(images[i])
        plt.title(f"Image {i+1}")
        
        # Draw bounding boxes
        for annotation in boxes[i]:
            if annotation[4] > 0:  # Check if occluded is greater than 0 (indicating occlusion)
                continue
            x1, y1, x2, y2 = annotation[:4]
            plt.gca().add_patch(plt.Rectangle((x1 * images.shape[1], y1 * images.shape[0]),
                                              (x2 - x1) * images.shape[1], (y2 - y1) * images.shape[0],
                                              edgecolor='red', facecolor='none', linewidth=2))
        
        plt.show()

    break 