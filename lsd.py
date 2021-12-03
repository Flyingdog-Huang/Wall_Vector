import cv2
import numpy as np
from pylsd.lsd import lsd

# load img data
path_read='./predict/'
# path_write='../do/'
img_name='1_img.png'
img=cv2.imread(path_read+img_name)
img_pre=cv2.imread(path_read+'1_pre.png')

# process img
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_canny=cv2.Canny(img_gray,70,150)
# print(img_gray)

# detect line
# HoughP_line
lines_houghp=cv2.HoughLinesP(img_canny,1,np.pi/180,80,30,10)
# LSD
img_dline=img_gray
lines_lsd=lsd(img_dline)
# Fast LSD
lsd_dec=cv2.ximgproc.createFastLineDetector()
lines_fastLSD=lsd_dec.detect(img_gray)

# draw line
img_houghp=img_pre.copy()
for line in lines_houghp:
    x1,y1,x2,y2=line[0]
    cv2.line(img_houghp, (x1,y1),(x2, y2),  (0,0,255), 1, cv2.LINE_AA)
img_lsd=img_pre.copy()
# print(dlines)
for dline in lines_lsd:
    x0, y0, x1, y1=map(int, dline[:4])
    cv2.line(img_lsd, (x0, y0), (x1,y1), (0,0,255), 1, cv2.LINE_AA)
img_fastLSD=img_pre.copy()
img_fastLSD=lsd_dec.drawSegments(img_fastLSD,lines_fastLSD)

cv2.imshow('gray',img_gray)
cv2.imshow('houghp',img_houghp)
cv2.imshow('fast LSD',img_fastLSD)
cv2.imshow('LSD',img_lsd)
cv2.waitKey(0)