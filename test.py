import cv2
import numpy as np

x=512
y=512
rec1=[[100,100],[100,250],[300,300],[200,20]]
rec1=np.array(rec1)[:,np.newaxis,:]
mask1=np.zeros((y,x),np.uint8)
cv2.drawContours(mask1,rec1[np.newaxis,:,:,:],0, (255), -1)

cv2.imshow('test',mask1)
cv2.waitKey(0)