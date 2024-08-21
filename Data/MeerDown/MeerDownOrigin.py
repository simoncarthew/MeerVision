import json
import pandas as pd
import glob
import os
import cv2
import shutil
import multiprocessing as mp

class MeerDownOrigin:
    def __init__(self, behaviour_file = "Data/MeerDown/origin/Annotations/behaviours.json", colour_file = "Data/MeerDown/origin/Annotations/behaviour_colours.json", annotations_folder = "Data/MeerDown/origin/Annotations", annotated_video_folder = "Data/MeerDown/origin/Annotated_videos"):

        # load in behaviours
        with open(behaviour_file, 'r') as file:
            self.behaviours = json.load(file)

        # load behaviour colors from the JSON file
        with open(colour_file, 'r') as file:
            self.behaviour_colors = json.load(file)
        
        # load the video files
        self.video_files = glob.glob(os.path.join(annotated_video_folder + '/area_1', '*.mp4')) + glob.glob(os.path.join(annotated_video_folder + '/area_2', '*.mp4'))

        # load the annotations
        annotations_df_list = []
        annotation_files = glob.glob(os.path.join(annotations_folder, '*.csv'))
        for file in annotation_files:
            df = pd.read_csv(file)
            df['video'] = os.path.basename(file).split('.')[0]
            if "C2" in file: df['area'] = 1 
            else: df['area'] = 2
            annotations_df_list.append(df)
        self.annotations = pd.concat(annotations_df_list, ignore_index=True)

    def create_annotated_frames(self, new_res = 0.65 , frame_rate = 30, annotations_save_path = 'Data/MeerDown/annotations.csv', frames_save_path = "Data/MeerDown/frames"):
        # reduce sampling period to 0.5 seonds
        reduced_annotations = self.annotations[self.annotations['frame_number'] % frame_rate == 0]

        # save annotations
        reduced_annotations.to_csv(annotations_save_path, sep=',', index=False, header=True, encoding='utf-8')

        # check if data exists and if they want to redo data
        redo = False
        if os.path.exists(frames_save_path):
            if input("Annotated frame images already exists, would you like to delete (y/n)? ") == "y":
                shutil.rmtree(frames_save_path)
                redo = True
        else:
            redo = True

        if redo:
            # create directory
            os.mkdir(frames_save_path)

            # Prepare for parallel processing
            num_workers = mp.cpu_count()  # Number of CPU cores
            pool = mp.Pool(processes=num_workers)

            # Create a list of arguments for the pool
            args = [(vid_path, frames_save_path, frame_rate, new_res, (640,640)) for vid_path in self.video_files]

            # Process videos in parallel
            pool.starmap(process_video, args)

            # Close and join the pool
            pool.close()
            pool.join()

            print("All videos have been processed.")

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

def process_video(vid_path, frames_path, frame_rate, new_res, image_size):
    # Get video name
    vid_name = os.path.basename(vid_path)[:-4]
    
    # Open the video file
    cap = cv2.VideoCapture(vid_path)

    # Check if file can be opened
    if not cap.isOpened():
        print("Error: Could not open " + vid_name)
        return

    # Set frame count
    frame_count = 0
    while True:
        # Read frame
        ret, frame = cap.read()

        # Check successful read or if end
        if not ret:
            break
        
        if frame_count % frame_rate == 0:
            # scale down the image resolution to 0.7 of its original size
            new_width = int(frame.shape[1] * new_res)
            new_height = int(frame.shape[0] * new_res)
            resized_frame = cv2.resize(frame, image_size)

            # Set file path
            image_path = os.path.join(frames_path, f"{vid_name}_frame_{frame_count}.jpg")

            # Save image
            cv2.imwrite(image_path, resized_frame)

        # Increase frame count
        frame_count += 1

    cap.release()
    print(f"Finished processing video: {vid_name}")

if __name__ == "__main__":
    md_data = MeerDownOrigin()
    md_data.create_annotated_frames(annotations_save_path="Data/MeerDown/reduced/annotations.csv", frames_save_path="Data/MeerDown/reduced/frames")