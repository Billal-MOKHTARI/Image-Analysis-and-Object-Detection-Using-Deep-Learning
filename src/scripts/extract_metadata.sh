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


echo "$metadata" >> "$output_csv"
exiftool -csv -q -r $folder_path > "$output_csv"

echo "Metadata extraction complete. Output saved to $output_csv"
