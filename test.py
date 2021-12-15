import cv2
import numpy as np
from pylsd.lsd import lsd
import math

path_read='./predict/'
img_name='1_mask.png'
img=cv2.imread(path_read+img_name)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for cnt in contours:
#     print(cnt)

print(type(contours))
print(type(contours[0]))
print(type(contours[0][0]))

'''
d=[4,3,2,1,4]

print(d.index(max(d)))


# load img data

# path_write='../do/'

img_pre=cv2.imread(path_read+'1_pre.png')

# process img


# LSD
img_dline=img_gray.copy()
lines_lsd=lsd(img_dline)

for dline in lines_lsd:
    print(dline)
    x0, y0, x1, y1=map(int, dline[:4])
    print(x0, y0, x1, y1)


x=512
y=512
# rec1=[[100,100],[100,250],[300,300],[200,20]]
# rec1=np.array(rec1)[:,np.newaxis,:]
mask1=np.zeros((y,x),np.uint8)
cv2.line(mask1, (0, 0), (600,600),255, 1)
# cv2.drawContours(mask1,rec1[np.newaxis,:,:,:],0, (255), -1)

cv2.imshow('test',mask1)
cv2.waitKey(0)
'''
# x1=10 
# x2=5
# for x in range(x1,x2,-1):
#     print(x)

# k=-1
# print(math.atan(k)*2/math.pi)

# a=80/180*math.pi
# print(math.tan(a))