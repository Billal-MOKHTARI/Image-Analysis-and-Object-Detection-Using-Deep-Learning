#!/bin/bash
source ../../configs/global.env

cd $EXIFTOOL_PATH

wget "https://exiftool.org/Image-ExifTool-12.81.tar.gz"


gzip -dc Image-ExifTool-12.81.tar.gz | tar -xf -
cd Image-ExifTool-12.81

perl Makefile.PL
make test

sudo make install