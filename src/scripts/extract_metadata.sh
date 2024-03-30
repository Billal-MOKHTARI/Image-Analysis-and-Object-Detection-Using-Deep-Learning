#!/bin/bash

# Validate input arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <folder_path> <output_csv>"
    exit 1
fi

folder_path="$1"
output_csv="$2"

# Check if folder exists
if [ ! -d "$folder_path" ]; then
    echo "Error: Folder '$folder_path' not found."
    exit 1
fi

# Extract metadata from images in the specified folder
echo "Extracting metadata from images in '$folder_path'..."
echo "File,Metadata" > "$output_csv"
for image_path in "$folder_path"/*
do
    echo "Processing $image_path..."
    if [ -f "$image_path" ]; then
        filename=$(basename "$image_path")
        metadata=$(exiftool -csv -q -r "$image_path")
        echo "$metadata" >> "$output_csv"
    fi
done

echo "Metadata extraction complete. Output saved to $output_csv"
