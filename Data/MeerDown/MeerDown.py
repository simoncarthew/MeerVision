import json
import pandas as pd
import glob
import os
import cv2
import shutil
import yaml

class MeerDown:

    def __init__(self, behaviour_file, colour_file, annotations_folder, annotated_video_folder):

        # load in behaviours
        with open(behaviour_file, 'r') as file:
            self.behaviours = json.load(file)

        # load behaviour colors from the JSON file
        with open(colour_file, 'r') as file:
            self.behaviour_colors = json.load(file)

        # load the annotations
        annotations_df_list = []
        annotation_files = glob.glob(os.path.join(annotations_folder, '*.csv'))
        for file in annotation_files:
            df = pd.read_csv(file)
            df['video'] = os.path.basename(file).split('.')[0]
            annotations_df_list.append(df)
        self.annotations = pd.concat(annotations_df_list, ignore_index=True)

        # load the video files
        self.video_files = glob.glob(os.path.join(annotated_video_folder + '/area_1', '*.mp4')) + glob.glob(os.path.join(annotated_video_folder + '/area_2', '*.mp4'))
    
    def visualise_annotations(self, video, video_folder):
        # filter annotations for video
        annotations = self.annotations[self.annotations['video'] == video]
        
        # Open the video file
        video_path = os.path.join(video_folder, video + ".mp4")
        cap = cv2.VideoCapture(video_path)

        # check if file can be opened
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        # initialise frame count
        frame_count = 0

        while True:
            # read frame
            ret, frame = cap.read()

            # check successful read
            if not ret:
                break
            
            # get relevant annotations
            frame_annotations = annotations[annotations['frame_number'] == frame_count]

            # draw annotation boxes
            for _, row in frame_annotations.iterrows():
                # get behaviour label
                behaviour_label = self.behaviours.get(str(row['behaviour_index']), 'Unknown')

                # get box colour
                color = self.behaviour_colors.get(str(row['behaviour_index']), (255, 255, 255))

                # draw bounding box
                cv2.rectangle(frame, (row['x1'], row['y1']), (row['x2'], row['y2']), color, 2)

                # add text label
                text = f"{behaviour_label}"
                cv2.putText(frame, text, (row['x1'], row['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            # display the frame
            cv2.imshow('Meerkats', frame)

            # Exit if the 'q' key is pressed
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
            frame_count += 1

        # Release the video capture object and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
    
    def create_yolo_dataset(self):
        # check if data exists and if they want to redo data
        redo = False
        yolo_path = "Data/MeerDown/Yolo"
        if os.path.exists(yolo_path):
            if input("Yolo data already exists, would you like to delete (y/n)? ") == "y":
                shutil.rmtree(yolo_path)
                redo = True
        else:
            redo = True

        # create new dataset
        if redo:
            # make the yolo folder
            os.makedirs(yolo_path, exist_ok=True)

            # format yaml file data
            data = {
                'path': 'Data/MeerDown/Yolo',
                'train': 'images/train',  
                'val': 'images/val',  
                'test': '',  
                'names': { 
                    0: 'meerkat',
                    }
                }
            
            # write data to yaml file
            file_path = 'Data/MeerDown/Yolo/data.yaml'
            with open(file_path, 'w') as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)

            # create relevant folders
            os.makedirs(yolo_path + '/images', exist_ok=True)
            os.makedirs(yolo_path + '/images/train', exist_ok=True)
            os.makedirs(yolo_path + '/images/val', exist_ok=True)
            os.makedirs(yolo_path + '/labels', exist_ok=True)
            os.makedirs(yolo_path + '/labels/train', exist_ok=True)
            os.makedirs(yolo_path + '/labels/val', exist_ok=True)

            # iterate over every video
            for vid_path in self.video_files:
                # get video name
                vid_name = os.path.basename(vid_path)[:-4]
                
                # open the video file
                cap = cv2.VideoCapture(vid_path)

                # check if file can be opened
                if not cap.isOpened():
                    print("Error: Could not open " + vid_name)
                    exit()

                # set frame count
                frame_count = 0

                while True: # iterate over the frames
                    # read frame
                    ret, frame = cap.read()

                    # check successful read
                    if not ret:
                        break
                    
                    # get the frame dimensions
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # get relevant annotations
                    frame_annotations = self.annotations[(self.annotations['frame_number'] == frame_count) & (self.annotations['video'] == vid_name)]

                    # iterate over annotations
                    lines = []
                    for _, row in frame_annotations.iterrows():
                        width = (int(row['x2']) - int(row['x1'])) / frame_width
                        height = (int(row['y2']) - int(row['y1'])) / frame_height
                        x_centre = width - width / 2
                        y_centre = height - height / 2
                        lines.append(f'0 {x_centre:.6f} {y_centre:.6f} {width:.6f} {height:.6f}')

                    # increase frame
                    frame_count += 1

                    # train or val
                    train_val = ""
                    if 'area_1' in vid_path:
                        train_val = "train"
                    else:
                        train_val = "val"

                    # set file paths
                    image_path = "Data/MeerDown/Yolo/images/" + train_val + '/' + vid_name + frame_count + ".jpg"
                    label_path = "Data/MeerDown/Yolo/label/" + train_val + '/' + vid_name + frame_count + ".txt"

                    # save image
                    cv2.imwrite(image_path)
                
                    # save annotations
                    with open(label_path, 'w') as file:
                        file.writelines(lines)
    
if __name__ == "__main__":
    md_data = MeerDown("Data/MeerDown/Annotations/behaviours.json","Data/MeerDown/Annotations/behaviour_colours.json","Data/MeerDown/Annotations", "Data/MeerDown/Annotated_videos")
    md_data.create_yolo_dataset()
    # print(md_data.annotations)
    # md_data.visualise_annotations("22-10-20_C2_06","Data/MeerDown/Annotated_videos/area_1")