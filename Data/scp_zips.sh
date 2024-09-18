#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <zip_directory> <remote_user> <remote_host> <remote_directory>"
    exit 1
fi

# Variables
ZIP_DIR=$1               # Directory containing the zip files
REMOTE_USER=$2           # Remote server username
REMOTE_HOST=$3           # Remote server hostname/IP
REMOTE_DIR=$4            # Remote directory to SCP to

# Check if the zip directory exists
if [ ! -d "$ZIP_DIR" ]; then
    echo "Error: Directory $ZIP_DIR does not exist."
    exit 1
fi

# SCP all zip files in the specified directory to the remote server
for zip_file in "$ZIP_DIR"/*.zip; do
    if [ -f "$zip_file" ]; then
        echo "Transferring $zip_file to $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
        scp "$zip_file" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
    else
        echo "No zip files found in $ZIP_DIR"
    fi
done

echo "All zip files transferred to $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
