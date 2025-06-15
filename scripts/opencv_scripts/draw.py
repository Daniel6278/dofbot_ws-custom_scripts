import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow('Blank', blank)

#paint img color
blank[200:300, 250:400] = 0,255,0
cv.imshow('Green', blank)

#draw recangle
cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=2)
cv.imshow('Rectangle', blank)

#draw circle
cv.circle(blank, (250,250), 40, (0,0,255), thickness=-1)
cv.imshow('Circle', blank)

#draw line
cv.line(blank, (0,0), (250,250), (255,0,0), thickness=2)
cv.imshow('Line', blank)

#draw text
cv.putText(blank, 'Hello World!', (0,255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), 2)
cv.imshow('Text', blank)
cv.waitKey(0)