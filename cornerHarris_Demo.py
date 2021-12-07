#!/home/fer/.virtualenvs/cv/bin/python
from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

source_window = 'Source image'
corners_window = 'Corners detected'
max_thresh = 255

def cornerHarris_demo(src_gray, val = 200):
    thresh = val

    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04

    ## cambio de variale
    src = np.array(src_gray, dtype = np.float32)

    # Detecting corners
    dst = cv.cornerHarris(src, blockSize, apertureSize, k)

    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)

    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                cv2.circle(dst_norm_scaled, (j,i), 5, (0), 2)

    # Showing the result
    cv.namedWindow(corners_window)
    cv.imshow(corners_window, dst_norm_scaled)

# Load source image and convert it to gray
parser = argparse.ArgumentParser(description='Code for Harris corner detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='opencv_frame_0.png')
args = parser.parse_args()
def main():
    src = cv.imread(cv.samples.findFile(args.input))
    if src is None:
        print('Could not open or find the image:', args.input)
        exit(0)

    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # Create a window and a trackbar
    cv.namedWindow(source_window)
    thresh = 200 # initial threshold
    cv.createTrackbar('Threshold: ', source_window, thresh, max_thresh, cornerHarris_demo)
    cv.imshow(source_window, src)
    cornerHarris_demo(src_gray, thresh)
    cv.waitKey()
main()
