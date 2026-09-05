# Simple program to load an image and display it
# Usage: python opencv.py [path/to/image.jpg]

import sys
import cv2

path = sys.argv[1] if len(sys.argv) > 1 else 'fri.jpg'
img = cv2.imread(path)

if img is None:
    print("Could not read image: " + path)
    print("Pass a path to any image, e.g. python opencv.py my_photo.jpg")
    sys.exit(1)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
