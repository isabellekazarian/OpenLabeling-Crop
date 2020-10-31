# OpenLabeling Crop
A simple script to crop images to their bounding boxes created in OpenLabeling.
This script is designed to use the YOLO_darknet output.

This script will process data for a single label only. It can, however, process multiple bounding boxes per image. For each bounding box, all crops will be made and saved with unique names.

## Prerequisites
* Python 3.7
* OpenCV
* numpy

## Getting Started
Check the paths of the following directories in the script:
* `INPUT_IMG_DIR`: Directory containing the input images. This should be the same folder that was opened in OpenLabeling.
* `BOUNDING_BOX_DIR`: Directory containing the bounding boxes. By default OpenLabeling will save to `\main\output\YOLO_darknet` in the OpenLabeling folder.
* `CROPPED_DIR`: Directory will be the destination for cropped images.

## Using the Script
Run the script.
Console output will show cropping statistics and identify images with no data.
All successfully cropped images will be saved to `CROPPED_DIR`

## References
[OpenLabeling](https://awesomeopensource.com/project/Cartucho/OpenLabeling)
