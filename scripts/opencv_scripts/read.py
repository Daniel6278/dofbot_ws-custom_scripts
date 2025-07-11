import cv2 as cv

img = cv.imread('Photos/cat.jpg')


def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
    
resized_img = rescaleFrame(img)
cv.imshow('Cat', resized_img)
#cv.imshow('Cat', img)
cv.waitKey(0)
