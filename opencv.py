#Simple program to image and read

import cv2
img = cv2.imread('fri.jpg')
cv2.imshow('Friends Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()