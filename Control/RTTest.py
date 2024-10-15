import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

MAN_COUNT_PATH = os.path.join("Control", "Results", "manual_detections")
DEP_PATH = os.path.join("Control", "Images", "Deployments", "2024_10_7_19_27_53")
PROC_PATH = os.path.join("Control", "Processed", "2024_10_7_19_27_53")
RES_PATH = os.path.join("Control", "Results")

FIGSIZE = (10,6)
TITLESIZE = 15
AXISSIZE = 12
TICKSIZE = 10
BARWIDTH = 0.3
COLOURS=[x for i,x in enumerate(matplotlib.colormaps.get_cmap('Set3')(np.linspace(0, 1, 10))) if i!=1]

# a function that make sure to add functionality to delete images with simon in them
def delte_images(folder_path):
    # List all JPEG files in the directory
    jpeg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.jpeg')]

    for file_name in jpeg_files:
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)
        
        if image is None:
            continue

        # Display image in fullscreen mode
        cv2.namedWindow('Image Viewer', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Image Viewer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Image Viewer', image)

        # Wait for user input
        key = cv2.waitKey(0)

        if key == 13:  # Enter key (Keep the image)
            print(f"Keeping: {file_name}")
        elif key == 127:  # Delete key (Delete the image)
            print(f"Deleting: {file_name}")
            os.remove(image_path)

        # Close the window before loading the next image
        cv2.destroyWindow('Image Viewer')

# a function to manually count the number of meerkats in each frame
def count_objects_in_images(folder_path, output_csv):
    # Keepin' track with pandas, yo!
    df = pd.DataFrame(columns=['image', 'actual_no_detections'])

    # Scan the JPEGs, one by one
    jpeg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.jpeg')]

    for file_name in jpeg_files:
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)

        if image is None:
            continue  # If it ain't there, we ain't messin' with it

        # Show it, big and bold
        cv2.namedWindow('Image Viewer', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Image Viewer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Image Viewer', image)

        print(f"Press a number key for {file_name} (0-9):")

        # Wait for a key press (up to 0-9 for object counts)
        key = cv2.waitKey(0)

        if 48 <= key <= 57:  # Keys '0' to '9'
            object_count = key - 48  # Convert from ASCII to integer
            print(f"Counted {object_count} objects.")
        else:
            print("Invalid input, setting count to 0.")
            object_count = 0

        # Save the image name and object count to DataFrame
        df.loc[len(df.index)] = [os.path.basename(file_name), object_count]

        # Close the window before loading the next image
        cv2.destroyWindow('Image Viewer')

    # Save it, cause that's the code
    df.to_csv(output_csv, index=False)

# a function taht computes the MSE between the actual number of meerkats and the 
# might just be one line
def compute_mse(list1, list2):
    # Ensure the two lists have the same length
    if len(list1) != len(list2):
        raise ValueError("Lists must be of equal length")
    
    # Compute MSE
    mse = np.mean((np.array(list1) - np.array(list2)) ** 2)
    return mse

# a function to plot th eactual number of meerkats vs predicted number of meerkats
def plot_with_mse(manual_path, proc_path, save_path):
    # read in df
    manual_df = pd.read_csv(manual_path)
    proc_df = pd.read_csv(proc_path)
    merged_df = pd.merge(manual_df, proc_df, on='image')
    print(merged_df)

    # Compute MSE
    mse = compute_mse(merged_df['no_detections'].values, merged_df['actual_no_detections'].values)

    # Plot both lines
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df["time"], merged_df['no_detections'].values, label="Predicted Count", color=COLOURS[0])
    plt.plot(merged_df["time"], merged_df['actual_no_detections'].values, label="Actual Count", color=COLOURS[1])
    
    # Add MSE as a text box in the plot
    plt.text(0.5, 0.1, f'MSE: {mse:.2f}', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5), fontsize=TITLESIZE)
    
    # Add labels and legend
    plt.xlabel('Time', fontsize=AXISSIZE)
    plt.ylabel('Number of Meerkats', fontsize=AXISSIZE)
    plt.xticks(rotation=45, fontsize=TICKSIZE)
    plt.ylim((0,max(max(merged_df['no_detections'].values), max(merged_df['actual_no_detections'].values)) +1))
    plt.title("Meerkat Count vs Time", fontsize=TITLESIZE)
    plt.legend()
    
    # Display the plot
    plt.savefig(save_path)

# delete images with simon
# delete_images(DEP_PATH)

# count objects in image
# count_objects_in_images(DEP_PATH, os.path.join(RES_PATH,"manual_counts.csv"))

# plot with mse
plot_with_mse(os.path.join(RES_PATH,"manual_counts.csv"), os.path.join(PROC_PATH,"meerkats_vs_time.csv"),save_path=os.path.join(RES_PATH,"res_path.png"))