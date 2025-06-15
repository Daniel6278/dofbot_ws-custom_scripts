
#basic.py
#basic cv2 image manipulation functions

import cv2 as cv

img = cv.imread('Photos/cat.jpg')
cv.imshow('Cat', img)

#convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#blur
#blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
#cv.imshow('Blur', blur)

#canny edge detection
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edges', canny)

#dilating the img
dilated = cv.dilate(canny, (3,3), iterations=1)
cv.imshow('Dilated', dilated)

#erode
eroded = cv.erode(dilated, (3,3), iterations=1)
cv.imshow('Eroded', eroded)

#resize
resize = cv.resize(img, (500,500))
cv.imshow('Resized', resize)

#cropping ie. array slicing of pixels
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)