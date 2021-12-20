import cv2
import numpy as np
from pylsd.lsd import lsd
import math

# load img data
path_read='./predict/'
# path_write='../do/'
img_name= '3.jpg' #  '1_img.png'
img=cv2.imread(path_read+img_name)
img_pre=cv2.imread(path_read+'3_pre_unet.png') # 1_pre


# process img
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Fast LSD
lsd_dec=cv2.ximgproc.createFastLineDetector()
lines_fastLSD=lsd_dec.detect(img_gray)

