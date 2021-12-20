import cv2
import numpy as np
from pylsd.lsd import lsd
import math

'''
path_read='./predict/'
img_name='1_mask.png'
img=cv2.imread(path_read+img_name,0)
# img_name1='1_pre_unet.png'
# img1=cv2.imread(path_read+img_name1,0)
y,x=img.shape
other_k_map=np.zeros((y,x),np.uint8)
other_k_map+=img
other_k_map+=img

cv2.imshow('other_k_map',other_k_map)


cv2.waitKey(0)
'''
# img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# contours, _ = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # for cnt in contours:
# #     print(cnt)

# print(type(contours))
# print(type(contours[0]))
# print(type(contours[0][0]))

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

# a=10/180*math.pi
# print(a)
# k=math.tan(a)
# print(k)
# print(math.atan(k))

# def is_K_same(k1,k2):
#     # 夹角在10°内
#     tan_a=(k1-k2)/(1+k1*k2+0.0001)
#     if tan_a**2<=math.tan(10/180*math.pi):
#         return True
#     return False

# k1=0.1
# k2=2
# print(is_K_same(k1,k2))

# for i in range(-5,6):
#     print(i)


img=np.zeros((512,512),np.uint8)
x1,y1=100,312
x2,y2=10,20
x3,y3=302,212
x4,y4=400,400
points=[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
points=np.array(points)
# cv2.line(img, (x1, y1), (x2,y2),255, 1)
# cv2.line(img, (x3, y3), (x4,y4),255, 1)
cv2.drawContours(img,[points],0,255,-1)

k_rg=0
rg_map=np.zeros((512,512),np.uint8)
rg_flag_map=np.zeros((512,512),np.uint8)


def isPointIn(point_x,point_y,x_b_min=0,x_b_max=512,y_b_min=0,y_b_max=512):
    #判断点是否在某一区域内
    if point_x>=x_b_min and point_x<x_b_max and point_y>=y_b_min and point_y<y_b_max:
        return True
    else:
        return False

def get_near_position(point_x,point_y,k_size=3):
    x_nears=[]
    y_nears=[]
    for i in range(k_size):
        for j in range(k_size):
            x_near=point_x+i-1
            y_near=point_y+j-1
            if isPointIn(x_near,y_near):
                x_nears.append(x_near)
                y_nears.append(y_near)
    return x_nears,y_nears

def is_k_point(point_x,point_y,x_nears,y_nears,k_stand,k_rg_map,k_rg_flag_map,thr=0.5):
    num_near=len(x_nears)
    print(point_y,point_x)
    k_rg_flag_map[point_y,point_x]=255 # 标记已经遍历
    # 是否属于K区域
    num_k=0
    for i in range(num_near):
        x_near,y_near=x_nears[i],y_nears[i]
        if img[y_near,x_near]-k_stand==0 :
            num_k+=1
            
    is_k_area=False
    if num_k>=int(num_near*thr):
        is_k_area=True
        k_rg_map[point_y,point_x]=255
    return is_k_area


def k_RG(point_x,point_y,k_stand,k_rg_map,k_rg_flag_map,k_size=3,thr=0.5):
    # 区域生长
    # 8邻域
    # 0.5
    # 判断是否已经遍历过
    if k_rg_flag_map[point_y,point_x]!=255:
        # k_rg_flag_map[point_y,point_x]=255 # 标记已经遍历
        x_nears,y_nears=get_near_position(point_x,point_y,k_size)

        seeds_x=[]
        seeds_y=[]
        is_k_area=is_k_point(point_x,point_y,x_nears,y_nears,k_stand,k_rg_map,thr)
        num_near=len(x_nears)
        
        # 初始化种子list
        if is_k_area:
            for i in range(num_near):
                x_near,y_near=x_nears[i],y_nears[i]
                near_x_nears,near_y_nears=get_near_position(x_near,y_near)
                if k_rg_flag_map[y_near,x_near]!=255 and is_k_point(x_near,y_near,near_x_nears,near_y_nears,k_stand,k_rg_map,thr):
                    seeds_x.append(x_near)
                    seeds_y.append(y_near)
        # 依据种子list开始生长
        while len(seeds_x)>0:
            seed_x=seeds_x.pop(0)
            seed_y=seeds_y.pop(0)
            seed_x_nears,seed_y_nears=get_near_position(seed_x,seed_y,k_size)
            num_near_near=len(seed_x_nears)
            for j in range(num_near_near):
                seed_x_near,seed_y_near=seed_x_nears[j],seed_y_nears[j]
                seed_near_x_nears,seed_near_y_nears=get_near_position(seed_x_near,seed_y_near)
                if k_rg_flag_map[seed_y_near,seed_x_near]!=255 and is_k_point(seed_y_near,seed_x_near,seed_near_x_nears,seed_near_y_nears,k_stand,k_rg_map,thr):
                    seeds_x.append(seed_x_near)
                    seeds_y.append(seed_y_near)


        

       
for x_rg in range(512):
    for y_rg in range(512):
        if rg_flag_map[y_rg,x_rg]!=255:
            k_RG(x_rg,y_rg,k_stand=k_rg,k_rg_map=rg_map,k_rg_flag_map=rg_flag_map)


# k1=(y1-y2)/(x1-x2)
# k2=-1/k1

# ld=30
# x0=(x1+x2)//2
# y0=(y1+y2)//2

# b1=y0-x0*k2
# dx=int((ld**2/(1+k2**2))**0.5)
# xd=int(x0+dx)
# yd=int(xd*k2+b1)

# img_d=np.zeros((512,512),np.uint8)
# cv2.line(img_d, (x0, y0), (xd,yd),255, 1)

# print(dx,xd,yd,x0, y0)

# for i in range(1,dx+1):
#     xi=x0+i
#     yi=int(xi*k2+b1)
#     for j in range(y0,yi+1):
#         if img_d[j,xi]==255 and img[j,xi]==255:
#             print(j,xi)
#             for ii in range(x0,xi+1):
#                 for jj in range(y0,yd+1):
#                     if img_d[jj,ii]==255:
#                         if jj!=y0 or ii!=x0:
#                             img[jj,ii]=255

        

cv2.imshow('line',img)
cv2.imshow('d-rg_map',rg_map)
cv2.waitKey(0)