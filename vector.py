import cv2
import numpy as np

def close_operate(img_b,k=5,it=3):
    # 膨胀=>腐蚀(闭运算)——充填空隙
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))# 定义核结构
    img_c=cv2.morphologyEx(img_b, cv2.MORPH_CLOSE, kernel, iterations=it)
    return img_c

def binary_process(img):
    img_g=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#灰化
    _,img_b=cv2.threshold(img_g,127,255,cv2.THRESH_BINARY)#二值化
    # 腐蚀=>膨胀(开运算)——去噪
    # img_o=cv2.morphologyEx(img_b, cv2.MORPH_OPEN, kernel, iterations=it)
    return img_b


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

def findCrossPoint(cnts):
    for cnt


def drawDT(img, subdiv,color):
    t_list= subdiv.getTriangleList()
    for t in t_list:
        print('t',t)
        p1=(t[0],t[1])
        p2=(t[2],t[3])
        p3=(t[4],t[5])
        cv2.line(img,p1,p2,color,2)
        cv2.line(img,p2,p3,color,2)
        cv2.line(img,p3,p1,color,2)

path= '../data/' # '../data/predict/'
name='1_mask.png'
img=cv2.imread(path+name)
img_orig = img.copy()
size = img.shape
rect = (0,0,size[1],size[0])

<<<<<<< HEAD
img_c, img_o=preprocess(img)
=======
name='1_pre.png'
img=cv2.imread('./predict/'+name)
img_b=binary_process(img)
img_c=close_operate(img_b)
>>>>>>> 5360e3ec11929e397792cbba2edd9401c509c00d

cnts=findConts(img_c)
# print(cnts)
# print(len(cnts))
cnts_f=filterConts(cnts)
# print(len(cnts_f))
'''
for cnt in cnts_f:
    
    print(cnt)
    print(cnt.shape)
    print(type(cnt))
    print(cv2.contourArea(cnt))

points=[[0,0],[1,0],[1,1],[0,1]]
points=np.array(points)
points=points[:,np.newaxis,:]
print(points.shape)
print(cv2.contourArea(points))



img_copy=img_orig.copy()
for cnt in cnts_f:
    print('cnt',cnt)
    subdiv = cv2.Subdiv2D(rect)
    for p in cnt:
        print('p',p)
        subdiv.insert(p)
        img_p_copy=img_orig.copy()
        drawDT(img_copy,subdiv,(0,0,255))
        drawDT(img_p_copy,subdiv,(0,255,0))
        # cv2.imshow("Delaunay Triangulation",img_p_copy)
        # cv2.waitKey(100)

cv2.imshow("Delaunay Triangulation",img_copy)

'''

rec1=[[100,100],[400,500],[440,470],[140,70]]
rec1=np.array(rec1)[:,np.newaxis,:]
img_test=img.copy()
cv2.drawContours(img_test,rec1[np.newaxis,:,:,:],0, (0,255,0), 1)
cv2.imshow('test',img_test)


# img_nf=img.copy()
# img_f=img.copy()
# cv2.drawContours(img_f,cnts_f,2, (0, 0, 255), 1)
# cv2.drawContours(img_nf,cnts,2, (0, 0, 255), 1)

# img_c=cv2.resize(img_c,(512,512))
# img_o=cv2.resize(img_o,(512,512))
# cv2.imshow('close',img_c)
# cv2.imshow('open',img_o)

# img_nf=cv2.resize(img_nf,(512,512))
# img_f=cv2.resize(img_f,(512,512))

# cv2.imshow('filter',img_f)
# cv2.imshow('no filter',img_nf)


cv2.waitKey(0)