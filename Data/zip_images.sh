#!/bin/bash

# Variables
IMAGE_DIR=$1                 # The directory containing the images
ZIP_PREFIX="images_part"      # Prefix for the zip files
NUM_ZIPS=10                  # Number of zip files to create
#REMOTE_USER=$2               # Remote server username
#REMOTE_HOST=$3               # Remote server hostname/IP
#REMOTE_DIR=$4                # Remote directory to SCP to
CURRENT_DIR=$(pwd)

# Check if correct number of arguments is passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <image_folder> <remote_user> <remote_host> <remote_directory>"
    exit 1
fi

# Count total number of images
TOTAL_IMAGES=$(ls -1q "$IMAGE_DIR"/*.{jpg,jpeg,png,gif} 2> /dev/null | wc -l)
if [ "$TOTAL_IMAGES" -eq 0 ]; then
    echo "No images found in $IMAGE_DIR"
    exit 1
fi

# Calculate how many images per zip file
IMAGES_PER_ZIP=$((TOTAL_IMAGES / NUM_ZIPS))
REMAINDER=$((TOTAL_IMAGES % NUM_ZIPS))

# Split images into groups and zip them
cd "$IMAGE_DIR" || exit
IMAGES=($(ls -1 *.jpg *.jpeg *.png *.gif))

START_INDEX=0
for ((i = 1; i <= NUM_ZIPS; i++)); do
    # Calculate the number of images for this zip file
    GROUP_SIZE=$IMAGES_PER_ZIP
    if [ "$i" -le "$REMAINDER" ]; then
        GROUP_SIZE=$((GROUP_SIZE + 1))
    fi

    # Create the list of images for this zip
    ZIP_FILE="$CURRENT_DIR/$ZIP_PREFIX$i.zip"
    echo "Creating $ZIP_FILE with $GROUP_SIZE images"
    zip "$ZIP_FILE" "${IMAGES[@]:$START_INDEX:$GROUP_SIZE}"
    
    # Update the start index for the next group
    START_INDEX=$((START_INDEX + GROUP_SIZE))
done

# SCP the zip files to the remote server
#for ZIP in "$ZIP_PREFIX"*.zip; do
#    echo "Transferring $ZIP to $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
#    scp "$ZIP" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
#done

#echo "All zip files transferred!"

# Optional: Clean up local zip files after SCP (uncomment if needed)
# rm "$ZIP_PREFIX"*.zip
