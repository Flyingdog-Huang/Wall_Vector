import cv2
import numpy as np
from pylsd.lsd import lsd

path_read='../../data/'
path_write='../do/'

img_name='pre1.png'
img=cv2.imread(path_read+img_name)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(img_gray)
# lsd=cv2.createLineSegmentDetector(0,scale=1)
dlines=lsd(img_gray)
# print(dlines)
for dline in dlines:
    x0, y0, x1, y1=map(int, dline[:4])
    cv2.line(img, (x0, y0), (x1,y1), (0,0,255), 3, cv2.LINE_AA)

cv2.imwrite(path_write+img_name,img)