import json
import pandas as pd
import glob
import os
import cv2

class MeerDown:

    def __init__(self, behaviour_file, colour_file, annotations_folder):

        # load in behaviours
        with open(behaviour_file, 'r') as file:
            self.behaviours = json.load(file)

        # Load behaviour colors from the JSON file
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
        
    
# Usage example
if __name__ == "__main__":
    # Creating an instance of MyClass
    md_data = MeerDown("Data/MeerDown/Annotations/behaviours.json","Data/MeerDown/Annotations/behaviour_colours.json","Data/MeerDown/Annotations")
    print(md_data.annotations)
    md_data.visualise_annotations("22-11-07_C3_03","Data/MeerDown/Annotated_videos")