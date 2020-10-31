import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# INPUT_IMG_DIR = r'C:\Users\path'
# BOUNDING_BOX_DIR = r'C:\Program Files (x86)\OpenLabeling\main\output\YOLO_darknet'
CROPPED_DIR = './images/cropped'


total_num_crops = 0
num_imgs_with_bb = 0
num_imgs_skipped = 0
skipped_imgs = []

for img_filename in os.listdir(INPUT_IMG_DIR):
    if img_filename.endswith(".jpg") or img_filename.endswith(".png"):

        # read image
        img = cv2.imread(INPUT_IMG_DIR + "/" + img_filename)
        print("Processing " + img_filename, end = '   ')
        img_height, img_width = img.shape[:2]

        # get bounding box filename
        bb_filename = img_filename[:-4] + '.txt'
        bb_file = BOUNDING_BOX_DIR + '\\' + bb_filename

        # check for valid data file
        if (not os.path.isfile(bb_file)) or (os.path.getsize(bb_file) == 0):
            num_imgs_skipped += 1
            skipped_imgs.append(bb_filename)
            print("No data")
            continue

        # process each bounding box in file
        # YOLO_darknet format: <object-class> <x_center> <y_center> <width> <height>
        bb_data = np.loadtxt(fname = bb_file, delimiter = " ", usecols=(1,2,3,4), ndmin=2)
        bb_num = 1

        for signature_bb in bb_data:
            x_center, y_center, width, height = signature_bb[:4]

            # get pixel data from proportional data
            bb_center = [(x_center * img_width),  (y_center * img_height)]
            bb_width  = int(width * img_width)
            bb_height = int(height * img_height)

            # find bounding corners
            bb_top_left = [int(bb_center[0] - (bb_width / 2)), int(bb_center[1] - (bb_height / 2))]
            bb_bottom_right = [int(bb_top_left[0] + bb_width), int(bb_top_left[1] + bb_height)]

            # crop & save
            cropped = img[bb_top_left[1]:bb_bottom_right[1], bb_top_left[0]:bb_bottom_right[0]]
            plt.imsave(CROPPED_DIR + '/' + img_filename[-19:-4] + '_' + str(bb_num) + '.png', cropped, cmap='gray')

            bb_num += 1  # generates unique filenames
            total_num_crops += 1   # counter

        print("Bounding boxes: " + str(bb_num))
        num_imgs_with_bb += 1


# output stats
print("\nFinished processing data.")
print("Cropped images saved to " + CROPPED_DIR)
print()
print(str(num_imgs_with_bb) + "  image(s) processed")
print(str(total_num_crops) + "  crop(s) made")
print()
print(str(num_imgs_skipped) + "  image(s) had no bounding box data:")

for image_name in skipped_imgs:
    print("  " + image_name)
