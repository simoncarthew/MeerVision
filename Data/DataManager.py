import json
import os
import glob
from pathlib import Path
import shutil
import math
import random
import cv2
from PIL import Image
import yaml

class DataManager():
    def __init__(self, perc_val = 0.2 , md_coco_path = "Data/MeerDown/raw/annotations.json", md_frames_path = "Data/MeerDown/raw/frames", obs_coco_path = "Data/Observed/annotations.json", obs_frames_path = "Data/Observed/frames", debug = True):
        # set class variables
        self.perc_val, self.debug = perc_val, debug
        self.obs_frames_path = obs_frames_path
        self.md_frames_path = md_frames_path

        # load meerdown annotations
        if os.path.exists(md_coco_path):
            with open(md_coco_path, 'r') as f:
                self.md_coco = json.load(f)

        # load observed annoations
        if os.path.exists(obs_coco_path):
            with open(obs_coco_path, 'r') as f:
                self.obs_coco = json.load(f)

    def view_coco_annotations(self, frame_folder, coco):
        # Load all frames from the frames folder
        frame_files = sorted(glob.glob(os.path.join(frame_folder, '*.jpg')))

        if not frame_files:
            print("No frames found in the specified folder.")
            return

        # Create a dictionary to map image IDs to annotations
        image_annotations = {}
        for img in coco["images"]:
            image_annotations[img["id"]] = []

        for ann in coco["annotations"]:
            image_id = ann["image_id"]
            if image_id in image_annotations:
                image_annotations[image_id].append(ann)

        # Create a named window and set it to full-screen
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        index = 0
        while index < len(coco["images"]):
            # Load and display the current frame
            frame_file = coco["images"][index]["file_name"]
            file_path = os.path.join(frame_folder,frame_file)
            frame = cv2.imread(file_path)

            if frame is None:
                print(f"Error loading frame: {frame_file}")
                break

            # Get the image ID for the current frame
            image_id =  coco["images"][index]["id"]

            # Draw bounding boxes on the frame
            if image_id in image_annotations:
                for ann in image_annotations[image_id]:
                    x, y, w, h = ann['bbox']
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

            # Show the frame in full-screen mode
            cv2.imshow('Frame', frame)

            # Wait for user input
            key = cv2.waitKey(0)

            # Move to the next or previous frame or exit
            if key == ord('m'):
                index = (index + 1) % len(frame_files)
            elif key == ord('n'):
                index = (index - 1) % len(frame_files)
            elif key == 27:  # ESC key
                break

        # Close the window
        cv2.destroyAllWindows()

    def view_yolo_annotations(self,images_folder, labels_folder, num_samples=5):
        # List all image files and annotation files
        image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]
        label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
        
        # Ensure there are images and labels to process
        if not image_files or not label_files:
            print("No images or labels found in the specified directories.")
            return
        
        # Randomly select images
        sample_images = random.sample(image_files, min(num_samples, len(image_files)))
        
        # Create a named window with fullscreen properties
        cv2.namedWindow('Image', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Display the selected images with annotations
        for image_file in sample_images:
            # Load image
            image_path = os.path.join(images_folder, image_file)
            image = cv2.imread(image_path)
            
            # Load corresponding annotation file
            label_file = image_file.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(labels_folder, label_file)
            
            if not os.path.exists(label_path):
                print(f"No annotation file for {image_file}. Skipping.")
                continue
            
            # Read annotations
            with open(label_path, 'r') as f:
                annotations = f.readlines()
            
            # Draw annotations on the image
            img_height, img_width = image.shape[:2]
            
            for annotation in annotations:
                parts = annotation.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert YOLO format to bounding box coordinates
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height
                
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Draw rectangle on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display the image
            cv2.imshow('Image', image)
            cv2.waitKey(0)  # Wait for a key press to move to the next image
        
        cv2.destroyAllWindows()

    def coco_to_yolo(self, coco, yolo_output_dir, train_val_test):

        # create directories if it doesnt exist
        Path(yolo_output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(yolo_output_dir,"labels",train_val_test)).mkdir(parents=True, exist_ok=True)
        
        # category to class mapping
        category_id_to_class_id = {category['id']: i for i, category in enumerate(coco['categories'])}
        
        # Iterate over the annotations
        for image in coco['images']:
            image_id = image['id']
            image_filename = image['file_name']
            image_width = image['width']
            image_height = image['height']
            
            # filter annotations for current image
            annotations = [ann for ann in coco['annotations'] if ann['image_id'] == image_id]
            
            # create yolo annotations
            yolo_annotations = []
            for ann in annotations:
                category_id = ann['category_id']
                bbox = ann['bbox']
                
                # convert bbox format
                x_min, y_min, width, height = bbox
                x_center = (x_min + width / 2) / image_width
                y_center = (y_min + height / 2) / image_height
                width /= image_width
                height /= image_height
                
                # get class id
                class_id = category_id_to_class_id[category_id]
                
                # add annotation
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # write annotations to text file
            yolo_label_path = os.path.join(yolo_output_dir, "labels", train_val_test, f"{Path(image_filename).stem}.txt")
            with open(yolo_label_path, 'w') as f:
                f.write("\n".join(yolo_annotations))

    def filter_md(self, md_z1_trainval_no, md_z2_trainval_no, md_test_no):
        # create zone 1 and zone 2 coco sets
        md_coco_1 = {"images": [], "annotations": [], "categories": self.md_coco["categories"]}
        md_coco_2 = {"images": [], "annotations": [], "categories": self.md_coco["categories"]}
        
        # populate images for each zone
        md_coco_1['images'] = [image for image in self.md_coco['images'] if image.get('zone') == 1]
        md_coco_2['images'] = [image for image in self.md_coco['images'] if image.get('zone') == 2]

        # get number of images
        num_z1_images = len(md_coco_1['images'])
        num_z2_images = len(md_coco_2['images'])

        # set sizes to full if -1
        if md_z1_trainval_no == -1: md_z1_trainval_no = num_z1_images
        if md_z2_trainval_no == -1: md_z2_trainval_no = num_z2_images

        # get sampling periods
        z1_samp_period = math.floor(num_z1_images / md_z1_trainval_no) 
        z2_samp_period = math.floor(num_z2_images / (md_z2_trainval_no + md_test_no))
        
        # check if sampling period is ok
        if z1_samp_period < 1:
            print("Invalid MeerDown zone 1 training size.")
            exit()
        if z2_samp_period < 1:
            print("Invalid MeerDown zone 2 training/test size.")
            exit()
            pass
        
        # sample images
        sampled_z1_images = [image for i, image in enumerate(md_coco_1['images']) if i % z1_samp_period == 0]
        sampled_z1_images = sampled_z1_images[0:md_z1_trainval_no]
        sampled_z2_images = [image for i, image in enumerate(md_coco_2['images']) if i % z2_samp_period == 0]
        sampled_z2_images = sampled_z2_images[0: md_z2_trainval_no + md_test_no]
        
        # get training images
        z1_train_end = math.floor(len(sampled_z1_images) * (1 - self.perc_val))
        z2_train_end = math.floor(md_z2_trainval_no * (1 - self.perc_val))
        train_images = sampled_z1_images[0:z1_train_end] + sampled_z2_images[0:z2_train_end]

        # get validation images
        val_images = sampled_z1_images[z1_train_end:] + sampled_z2_images[z2_train_end:md_z2_trainval_no]

        # get test images
        test_images = sampled_z2_images[md_z2_trainval_no:]

        # create training, validation, and test coco sets
        md_train = {"images": [], "annotations": [], "categories": self.md_coco["categories"]}
        md_val = {"images": [], "annotations": [], "categories": self.md_coco["categories"]}
        md_test = {"images": [], "annotations": [], "categories": self.md_coco["categories"]}

        # get images
        md_train['images'] = train_images
        md_val['images'] = val_images
        md_test['images'] = test_images

        # get annotations
        train_image_ids = {image['id'] for image in train_images}
        val_image_ids = {image['id'] for image in val_images}
        test_image_ids = {image['id'] for image in test_images}

        md_train['annotations'] = [ann for ann in self.md_coco['annotations'] if ann['image_id'] in train_image_ids]
        md_val['annotations'] = [ann for ann in self.md_coco['annotations'] if ann['image_id'] in val_image_ids]
        md_test['annotations'] = [ann for ann in self.md_coco['annotations'] if ann['image_id'] in test_image_ids]

        # check for overlapping images
        overlap_train_val = train_image_ids.intersection(val_image_ids)
        overlap_train_test = train_image_ids.intersection(test_image_ids)
        overlap_val_test = val_image_ids.intersection(test_image_ids)
        if overlap_train_val:
            print(f"Warning: Overlapping images between training and validation sets: {overlap_train_val}")
        if overlap_train_test:
            print(f"Warning: Overlapping images between training and test sets: {overlap_train_test}")
        if overlap_val_test:
            print(f"Warning: Overlapping images between validation and test sets: {overlap_val_test}")

        return md_train, md_val, md_test
    
    def filter_observed(self, obs_no):
        # test images
        test_images = [image for image in self.obs_coco['images'] if image.get('camera_trap') == True]

        # extract images from annotations
        non_test_images = [image for image in self.obs_coco['images'] if image.get('camera_trap') == False]

        # set sizes to full if -1
        if obs_no == -1: obs_no = len(non_test_images) 

        # ensure there are enough observations
        if len(non_test_images) < obs_no:
            print("Requested too many observation images")
            exit()

        # randomly sample total images
        sampled_images = random.sample(non_test_images, obs_no)

        # randomly sample validation images
        num_val_images = int(obs_no * self.perc_val)
        val_images = random.sample(sampled_images, num_val_images)

        # get training images
        train_images = [img for img in sampled_images if img not in val_images]

        # final coco sets
        obs_train = {"images": [], "annotations": [], "categories": self.obs_coco.get("categories", [])}
        obs_val = {"images": [], "annotations": [], "categories": self.obs_coco.get("categories", [])}
        obs_test = {"images": [], "annotations": [], "categories": self.obs_coco.get("categories", [])}

        # populate coco sets with images
        obs_train['images'] = train_images
        obs_val['images'] = val_images
        obs_test['images'] = test_images

        # extract the image ids
        train_image_ids = {image['id'] for image in train_images}
        val_image_ids = {image['id'] for image in val_images}
        test_image_ids = {image['id'] for image in test_images}

        # Populate annotations for each dataset
        obs_train['annotations'] = [ann for ann in self.obs_coco.get('annotations', []) if ann['image_id'] in train_image_ids]
        obs_val['annotations'] = [ann for ann in self.obs_coco.get('annotations', []) if ann['image_id'] in val_image_ids]
        obs_test['annotations'] = [ann for ann in self.obs_coco.get('annotations', []) if ann['image_id'] in test_image_ids]

        return obs_train, obs_val, obs_test

    def merge_coco(self, coco1, coco2):
        # initialize merged coco
        merged_coco = {
            'images': [],
            'annotations': [],
            'categories': coco1['categories'] 
        }

        # get the total number of images in both cocos
        coco1_size = len(coco1['images'])
        coco2_size = len(coco2['images'])

        # create global image ids
        global_image_ids = list(range(coco1_size + coco2_size))

        # create image maps
        image_id_map_c1 = {}
        image_id_map_c2 = {}

        # update image id of coco1
        for i, img in enumerate(coco1['images']):
            global_id = global_image_ids[i]
            image_id_map_c1[img['id']] = global_id
            img['id'] = global_id
            merged_coco['images'].append(img)

        # update annotations of coco1
        for ann in coco1['annotations']:
            ann['image_id'] = image_id_map_c1[ann['image_id']]
            merged_coco['annotations'].append(ann)

        # update image id of coco2
        for i, img in enumerate(coco2['images']):
            global_id = global_image_ids[coco1_size + i]
            image_id_map_c2[img['id']] = global_id
            img['id'] = global_id
            merged_coco['images'].append(img)

        # update annotations of coco2
        for ann in coco2['annotations']:
            ann['image_id'] = image_id_map_c2[ann['image_id']]
            merged_coco['annotations'].append(ann)
        
        return merged_coco

    def merge_images(self, coco1, coco2, images1_folder, images2_folder, destination_folder, img_size=None):
        # Create destination folder
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)
        os.makedirs(destination_folder)

        def copy_and_resize_image(image_file, source_folder, dest_folder, img_size):
            source_path = os.path.join(source_folder, image_file)
            dest_path = os.path.join(dest_folder, image_file)

            if img_size is not None:
                with Image.open(source_path) as img:
                    if img.size != img_size:
                        # Convert to RGB if necessary
                        if img.mode == 'RGBA':
                            img = img.convert('RGB')
                        img = img.resize(img_size, Image.LANCZOS)
                        img.save(dest_path)
                    else:
                        # If the image size matches, copy without resizing
                        shutil.copy(source_path, dest_path)
            else:
                # Just copy the image without resizing
                shutil.copy(source_path, dest_path)

        # Copy and resize images from the first dataset
        image_ids1 = {img['file_name'] for img in coco1['images']}
        for image_file in os.listdir(images1_folder):
            if image_file in image_ids1:
                copy_and_resize_image(image_file, images1_folder, destination_folder, img_size)

        # Copy and resize images from the second dataset
        if coco2 is not None and images2_folder is not None:
            image_ids2 = {img['file_name'] for img in coco2['images']}
            for image_file in os.listdir(images2_folder):
                if image_file in image_ids2:
                    copy_and_resize_image(image_file, images2_folder, destination_folder, img_size)

    def create_yolo_dataset(self, obs_no, md_z1_trainval_no, md_z2_trainval_no, md_test_no, yolo_path, img_size = (640,640)):
        # delete the existing output directory if it exists
        if os.path.exists(yolo_path):
            shutil.rmtree(yolo_path)

        # filter datasets
        obs_train, obs_val, obs_test = self.filter_observed(obs_no)
        if self.debug: print("Filtered observed.")
        md_train, md_val, md_test = self.filter_md(md_z1_trainval_no, md_z2_trainval_no, md_test_no)
        if self.debug: print("Filtered MeerDown.")

        # merge datasets
        train = self.merge_coco(obs_train, md_train)
        val = self.merge_coco(obs_val, md_val)
        test = self.merge_coco(obs_test, md_test)
        if self.debug: print("Mereged datasets.")

        # convert coco to yolo
        self.coco_to_yolo(train,yolo_path,"train")
        if self.debug: print("Training labels created")
        self.coco_to_yolo(val,yolo_path,"val")
        if self.debug: print("Validation labels created")
        self.coco_to_yolo(test,yolo_path,"test")
        if self.debug: print("Test labels created")

        # create image folders
        Path(os.path.join(yolo_path,"images","train")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(yolo_path,"images","val")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(yolo_path,"images","test")).mkdir(parents=True, exist_ok=True)

        # store images
        self.merge_images(obs_train,md_train,self.obs_frames_path,self.md_frames_path,os.path.join(yolo_path,"images","train"),img_size=img_size)
        if self.debug: print("Copied training images")
        self.merge_images(obs_val,md_val,self.obs_frames_path,self.md_frames_path,os.path.join(yolo_path,"images","val"),img_size=img_size)
        if self.debug: print("Copied validation images")
        self.merge_images(md_test,obs_test,"Data/MeerDown/raw/frames","Data/Observed/frames",os.path.join(yolo_path,"images","test"),img_size=img_size)
        if self.debug: print("Copied test images")

        # Create the .yaml file
        yaml_content = {
            'path': os.path.abspath(yolo_path),
            'nc':1,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {0: 'meerkat'}
        }
        with open(os.path.join(yolo_path, 'dataset.yaml'), 'w') as yaml_file:
            yaml.dump(yaml_content, yaml_file, default_flow_style=False)
        
    def create_dataloaders(self):
        pass

if __name__ == "__main__":
    dm = DataManager()
    dm.create_yolo_dataset(-1,1000,1000,0,"Data/Formated/yolo")
    dm.create_yolo_dataset()
    # dm.view_yolo_annotations("Data/Formated/yolo/images/test","Data/Formated/yolo/labels/test",10)

    # test filter_obsserved
    # obs_train, obs_val, obs_test = dm.filter_observed(-1)
    # dm.view_coco_annotations("Data/Observed/frames",obs_test)
    # print("Number of obs training images: " + str(len(obs_train["images"])))
    # print("Number of obs validation images: " + str(len(obs_val["images"])))
    # print("Number of obs testing images: " + str(len(obs_test["images"])))

    # test filter_md
    # md_train, md_val, md_test = dm.filter_md(1000,1000,0)
    # print("Number of md training images: " + str(len(md_train["images"])))
    # print("Number of md validation images: " + str(len(md_val["images"])))
    # print("Number of md testing images: " + str(len(md_test["images"])))

    
