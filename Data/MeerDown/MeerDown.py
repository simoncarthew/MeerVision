import glob
import pandas as pd
import os
import json
import multiprocessing as mp
import cv2

class MeerDown():
    def __init__(self, annotations_folder, videos_folder, output_folder, sampling_rate=30):
        self.annotations_folder = annotations_folder
        self.videos_folder = videos_folder
        self.output_folder = output_folder
        self.sampling_rate = sampling_rate
        
        # Create the annotations DataFrame
        self.annotations = self.create_annotations_df()
        print("Created annotations df.")
        
        # Get video file paths
        self.video_files = self.get_video_files()
        print("Loaded video files.")
        
        # create or load annotations folder
        coco_path = os.path.join(self.output_folder,"annotations.json")
        self.annot_exists = False
        if os.path.exists(coco_path):
            with open(coco_path, 'r') as f:
                self.coco = json.load(f)
            self.annot_exists = True
        else:
            self.coco = {
                "images": [],
                "annotations": [],
                "categories": [
                    {
                        "id": 1,
                        "name": "meerkat"
                    }
                ]
            }

        # set image dimensions 
        self.set_dimensions()

        # check if frames folder exists
        frame_path = os.path.join(output_folder,"frames")
        self.frame_exists = False
        if os.path.exists(frame_path):
            self.frame_exists = True

    def create_annotations_df(self):
        annotations_df_list = []
        annotation_files = glob.glob(os.path.join(self.annotations_folder, '*.csv'))

        for file in annotation_files:
            df = pd.read_csv(file)
            df['video'] = os.path.basename(file).split('.')[0]
            df['area'] = 1 if "C2" in file else 2
            annotations_df_list.append(df)

        return pd.concat(annotations_df_list, ignore_index=True)

    def get_video_files(self):
        return (glob.glob(os.path.join(self.videos_folder + '/area_1', '*.mp4')) +
                glob.glob(os.path.join(self.videos_folder + '/area_2', '*.mp4')))

    def set_dimensions(self):
        video_name = os.path.basename(self.video_files[0]).split('.')[0]
        cap = cv2.VideoCapture(self.video_files[0])

        # Retrieve video height and width
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def extract_frames(self, video_file):
        video_name = os.path.basename(video_file).split('.')[0]
        cap = cv2.VideoCapture(video_file)
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % self.sampling_rate == 0:
                # Generate image file name
                image_name = f"{video_name}_frame_{frame_id}.jpg"
                image_path = os.path.join(self.output_folder, "frames", image_name)

                # Save the frame as an image
                cv2.imwrite(image_path, frame)

            frame_id += 1

        cap.release()

    def create_coco_annotations(self):
        do = True
        if self.annot_exists:
            do = False
            if input("Coco annotations already exist. Would you like to regenerate them? (y/n)") == "y":
                do = True

        if do:
            print("Creating coco annotations.")

            # Initialize image and annotation IDs
            image_id = 0
            annotation_id = 0

            for video_file in self.video_files:
                # Get video name 
                video_name = os.path.basename(video_file).split('.')[0]

                # Filter annotations for the specific video
                annotations_filt = self.annotations[self.annotations['frame_number'] % self.sampling_rate == 0]
                annotations_filt = annotations_filt[annotations_filt['video'] == video_name]
                
                # Track the last frame number to identify new frames
                frame_no = None

                for _, row in annotations_filt.iterrows():
                    # Add image if new frame
                    if int(row['frame_number']) != frame_no:
                        frame_no = row['frame_number']
                        
                        # Add image to coco
                        image_info = {
                            "id": image_id,
                            "file_name": f"{video_name}_frame_{frame_no}.jpg",
                            "height": self.height,
                            "width": self.width,
                            "zone": 1 if "C2" in video_name else 2
                        }
                        self.coco["images"].append(image_info)
                        
                        # Increment image ID
                        image_id += 1

                    # Add annotation
                    annotation_info = {
                        "id": annotation_id,
                        "image_id": image_id - 1,  # Use the most recent image ID
                        "category_id": 1,
                        "bbox": [row['x1'], row['y1'], row['x2'] - row['x1'], row['y2'] - row['y1']],
                        "area": (row['x2'] - row['x1']) * (row['y2'] - row['y1']),
                        "iscrowd": 0
                    }
                    self.coco["annotations"].append(annotation_info)
                    
                    # Increment annotation ID
                    annotation_id += 1

                print("Completed " + video_name + " annotations.")

    def save_coco_file(self):
        with open(os.path.join(self.output_folder, "annotations.json"), 'w') as f:
            json.dump(self.coco, f, indent=4)

    def process_videos(self):
        do = True
        if self.frame_exists:
            do = False
            if input("Frames already exist. Would you like to regenerate them? (y/n)") == "y":
                do = True

        if do:
            # Create output folder if it doesn't exist
            os.makedirs(self.output_folder, exist_ok=True)
            
            # Use multiprocessing to extract frames from videos in parallel
            with mp.Pool(mp.cpu_count()) as pool:
                pool.map(self.extract_frames, self.video_files)

            # Create annotations for COCO format
            self.create_coco_annotations()

            # Save the final COCO file
            self.save_coco_file()

    def view_annotations(self):
        # Load all frames from the frames folder
        frame_files = sorted(glob.glob(os.path.join(self.output_folder, "frames", '*.jpg')))

        if not frame_files:
            print("No frames found in the specified folder.")
            return

        # Create a dictionary to map image IDs to annotations
        image_annotations = {}
        for img in self.coco["images"]:
            image_annotations[img["id"]] = []

        for ann in self.coco["annotations"]:
            image_id = ann["image_id"]
            if image_id in image_annotations:
                image_annotations[image_id].append(ann)

        # Initialize index for frame navigation
        index = 0

        while True:
            # Load and display the current frame
            frame_file = frame_files[index]
            frame = cv2.imread(frame_file)

            if frame is None:
                print(f"Error loading frame: {frame_file}")
                break

            # Get the image ID for the current frame
            frame_name = os.path.basename(frame_file)
            image_id = next((img["id"] for img in self.coco["images"] if img["file_name"] == frame_name), None)

            if image_id is None:
                print(f"No image ID found for frame: {frame_name}")
                break

            # Draw bounding boxes on the frame
            if image_id in image_annotations:
                for ann in image_annotations[image_id]:
                    x, y, w, h = ann['bbox']
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

            # Show the frame
            cv2.imshow('Frame', frame)

            # Wait for user input
            key = cv2.waitKey(0)

            # Move to the next or previous frame or exit
            if key == ord('m'):
                index = (index + 1) % len(frame_files)
            elif key == ord('n'):
                index = (index - 1) % len(frame_files)
            elif key == 27:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    md = MeerDown("Data/MeerDown/origin/Annotations","Data/MeerDown/origin/Annotated_videos","Data/MeerDown/raw")
    md.create_coco_annotations()
    md.save_coco_file()
    # md.view_annotations()