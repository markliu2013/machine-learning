# 例子，框处破损处
import numpy as np
import cv2 as cv

img_path = '../data/6e7d123148439d3a0931229603.jpg'

img = cv.imread(img_path,0)
d1 = img[500:547, 2171:2190]
cv.rectangle(img,(2171,500),(2190,547),(0,255,0),3)
cv.rectangle(img,(1241,448),(1255,487),(0,255,0),3)
# cv.imshow('image', img)
cv.imwrite('01.jpg',img)
cv.waitKey(0)
cv.destroyAllWindows()