#!/bin/bash

DOWNLOAD_LINK = "https://exiftool.org/Image-ExifTool-12.81.tar.gz"

wget $1
cd $1

gzip -dc Image-ExifTool-12.81.tar.gz | tar -xf -
cd Image-ExifTool-12.81

