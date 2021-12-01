import cv2
import numpy as np

def preprocess(img):
    img_g=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#灰化
    ret,img_b=cv2.threshold(img_g,127,255,cv2.THRESH_BINARY)#二值化
    k=5
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))# 定义核结构
    it=3
    # 膨胀=>腐蚀(闭运算)——充填空隙
    img_c=cv2.morphologyEx(img_b, cv2.MORPH_CLOSE, kernel, iterations=it)
    # 腐蚀=>膨胀(开运算)——去噪
    img_o=cv2.morphologyEx(img_b, cv2.MORPH_OPEN, kernel, iterations=it)
    return img_c, img_o


def findConts(img):
    '''
    find contours pionts
    '''
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filterConts(contours, eps=1):
    '''
    filter contours pionts
    '''
    num=len(contours)

    # 面积过滤——剔除杂波
    total_area=0
    for cnt in contours:
        total_area+=cv2.contourArea(cnt)
    contours_new=[]
    for i in range(num):
        cnt=contours[i]
        area=cv2.contourArea(cnt)
        if area>0.001*total_area:
            contours_new.append(cnt)
    
    # 更新
    contours=contours_new
    num=len(contours)

    # 轮廓近似
    for i in range(num):
        cnt=contours[i]
        # print(cv2.isContourConvex(cnt)) # 判断形状是否为凸
        if not eps:
            eps=0.001*cv2.arcLength(cnt,True)
        cnt=cv2.approxPolyDP(cnt, eps,True)
        contours[i]=cnt
    
    return contours


name='1_pre.png'
img=cv2.imread('../../data/'+name)
img_c, img_o=preprocess(img)

''''''
cnts=findConts(img_c)
print(len(cnts))
cnts_f=filterConts(cnts)
print(len(cnts_f))

img_nf=img.copy()
img_f=img.copy()
cv2.drawContours(img_f,cnts_f,-1, (0, 0, 255), 3)
cv2.drawContours(img_nf,cnts,-1, (0, 0, 255), 3)

img_c=cv2.resize(img_c,(512,512))
img_o=cv2.resize(img_o,(512,512))
cv2.imshow('close',img_c)
cv2.imshow('open',img_o)
img_nf=cv2.resize(img_nf,(512,512))
img_f=cv2.resize(img_f,(512,512))
cv2.imshow('filter',img_f)
cv2.imshow('no filter',img_nf)


cv2.waitKey(0)