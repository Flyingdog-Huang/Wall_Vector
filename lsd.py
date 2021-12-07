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
# img_canny=cv2.Canny(img_gray,70,150)
# print(img_gray)

# detect line
# # HoughP_line
# lines_houghp=cv2.HoughLinesP(img_canny,1,np.pi/180,80,30,10)
# LSD
# img_dline=img_gray.copy()
# lines_lsd=lsd(img_dline)

# Fast LSD
lsd_dec=cv2.ximgproc.createFastLineDetector()
lines_fastLSD=lsd_dec.detect(img_gray)

# # draw line
# img_houghp=img_pre.copy()
# for line in lines_houghp:
#     x1,y1,x2,y2=line[0]
#     cv2.line(img_houghp, (x1,y1),(x2, y2),  (0,0,255), 1, cv2.LINE_AA)
# img_lsd=img_pre.copy()
# # print(dlines)
# for dline in lines_lsd:
#     x0, y0, x1, y1=map(int, dline[:4])
#     cv2.line(img_lsd, (x0, y0), (x1,y1), (0,0,255), 1, cv2.LINE_AA)
# img_fastLSD=img_pre.copy()
# img_fastLSD=lsd_dec.drawSegments(img_fastLSD,lines_fastLSD)

# filter lines
# print(lines_fastLSD.shape)
img_wall=img_pre.copy()
# print(img_wall.shape)

# get BINARY prediction img
img_bool=img_pre.copy()
img_bool=cv2.cvtColor(img_bool, cv2.COLOR_BGR2GRAY)#灰化
print(img_bool.shape)
_,img_bool=cv2.threshold(img_bool,127,255,cv2.THRESH_BINARY)#二值化
print(img_bool.shape)
# process bool img : 膨胀=>腐蚀(闭运算)——充填空隙
k_dila=5
k_ero=3
it_dila=3
it_ero=1
# 定义核结构
kernel_dila=cv2.getStructuringElement(cv2.MORPH_RECT,(k_dila,k_dila))
kernel_ero=cv2.getStructuringElement(cv2.MORPH_RECT,(k_ero,k_ero))
# 膨胀
img_bool=cv2.dilate(img,kernel_dila,it_dila)
# 腐蚀
img_bool=cv2.erode(img,kernel_ero,it_ero)

# kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
# img_c=cv2.morphologyEx(img_b, cv2.MORPH_CLOSE, kernel, iterations=it)
img_bool=cv2.cvtColor(img_bool, cv2.COLOR_BGR2GRAY)#灰化
_,img_bool=cv2.threshold(img_bool,127,1,cv2.THRESH_BINARY)#二值化
print(img_bool)
print(img_bool.shape)
# get img size
y,x=img_bool.shape

# save wall lines , direction and k**2
wall_lines=[]
direc_lines=[]
k_lines=[]

# process condition
a1=2 # fusion dis
a2=8 # detection dis

# init detection position
dx=0
dy=0

# wall area condition
thr_mp=0.01 # exist condition
thr_dis=0.2 # diff condition

# K feature map
K_map=np.zeros((y,x,3),np.uint8)

# k max init
k_max=0

# max color
max_color=255+255

# filter wall lines
for line in lines_fastLSD:
    # print(line[0])
    x1,y1,x2,y2=line[0]
    if (x1-x2)**2<=a1**2 and (y1-y2)**2<=a1**2:
        continue

    # if (x1-x2)**2+(y1-y2)**2>50**2:

    x0,x3,x4,x5,=x1,x1,x1,x1
    y0,y3,y4,y5,=y1,y1,y1,y1
    if (x1-x2)**2<=a1**2:
        # x1=int(x1)
        x2=x1
        dx=a2
        dy=0
    elif (y1-y2)**2<=a1**2:
        # y1=int(y1)
        y2=y1
        dx=0
        dy=a2
    else:
        k=(x1-x2)/(y2-y1)
        dx=((a2**2)/(1+k**2))**0.5
        dy=dx*k
    
    x5=x1-dx 
    y5=y1-dy
    x4=x2-dx
    y4=y2-dy

    x3=x1+dx
    y3=y1+dy
    x0=x2+dx
    y0=y2+dy
    
    x0,x1,x2,x3,x4,x5,y0,y1,y2,y3,y4,y5=int(x0),int(x1),int(x2),int(x3),int(x4),int(x5),int(y0),int(y1),int(y2),int(y3),int(y4),int(y5)

    # filter line
    rec1=[[x1,y1],[x2,y2],[x4,y4],[x5,y5]]
    rec2=[[x1,y1],[x2,y2],[x0,y0],[x3,y3]]
    cnts=[]
    cnts.append(rec1)
    cnts.append(rec2)
    rec1=np.array(rec1)[:,np.newaxis,:]
    rec2=np.array(rec2)[:,np.newaxis,:]
    # print(rec1.shape)
    # area1=cv2.contourArea(rec1)
    # area2=cv2.contourArea(rec2)
    # print(area1,area2)
    # create mask
    mask1=np.zeros((y,x),np.uint8)
    # print(np.sum(mask1))
    cv2.drawContours(mask1,rec1[np.newaxis,:,:,:],0, (255), -1)
    # print(np.sum(mask1)/255)
    mask2=np.zeros((y,x),np.uint8)
    # print(np.sum(mask2))
    cv2.drawContours(mask2,rec2[np.newaxis,:,:,:],0, (255), -1)
    # mask1_area=np.sum(mask1*img_bool)
    # mask2_area=np.sum(mask2*img_bool)
    # mask1_p=mask1_area/np.sum(mask1)
    # mask2_p=mask2_area/np.sum(mask2)

    # print(np.sum(mask2)/255)
    _,mask1_b=cv2.threshold(mask1,127,1,cv2.THRESH_BINARY)#二值化
    _,mask2_b=cv2.threshold(mask2,127,1,cv2.THRESH_BINARY)#二值化
    # print(np.sum(img_bool),np.sum(mask1_b),np.sum(mask2_b))
    mask1_area=np.sum(mask1_b*img_bool)
    mask2_area=np.sum(mask2_b*img_bool)
    mask1_p=mask1_area/np.sum(mask1_b)
    mask2_p=mask2_area/np.sum(mask2_b)

    if (mask1_p+mask2_p)>(thr_mp*2) and (mask1_p-mask2_p)**2>(thr_dis**2): # True:
        # add wall lines
        wall_lines.append(line)
        print('wall-line',line)
        direc_line=[x1,y1,x5,y5] if mask1_p>mask2_p else [x1,y1,x3,y3]
        direc_lines.append(direc_line)
        # print(direc_line)
        k_2=-1
        if x1==x2:
            k_2=max_color
        elif y1==y2:
            k_2=0
        else:
            k_2=(line[0][1]-line[0][3])/(line[0][0]-line[0][2])  # (y1-y2)/(x1-x2)
            k_2=k_2**2
        # print(k_2)
        if k_2>max_color:
            k_2=max_color
        if k_2<0.00001:
            k_2=0
        # if k_2<999999:
        #     k_max=max(k_2,k_max)
        k_lines.append(k_2)
        # print(k_max)

        # bool wall direction
        t1=1 if mask2_p>mask1_p else -1
        t2=1 if mask1_p>mask2_p else -1
        
        # k color
        a_k_color=int(k_2)
        print('K',a_k_color)

        k_b,k_g,k_r=255,0,0
        
        i_color=a_k_color//255
        num_color=a_k_color-i_color*255
        
        if i_color==0:
            k_g=num_color
        if i_color==1:
            k_g=255
        if i_color==2:
            k_g,k_r=255,255
        k_color=(k_b,k_g,k_r)
        print('k_color',k_color)

        # draw k feature map
        if t1==-1:
            cv2.drawContours(K_map,rec1[np.newaxis,:,:,:],0, k_color, t1)
        else:
            cv2.drawContours(K_map,rec2[np.newaxis,:,:,:],0, k_color, t2)


        # draw wall direction area
        cv2.drawContours(img_wall,rec1[np.newaxis,:,:,:],0, (0,255,0), t1)
        cv2.drawContours(img_wall,rec2[np.newaxis,:,:,:],0, (0,255,0), t2)

        # draw line
        cv2.line(img_wall, (x2, y2), (x1,y1), (0,0,255), 2, cv2.LINE_AA)
        
        print('-------------------')


    
    # cnts=np.array(cnts)
    # cv2.drawContours(img_wall,cnts[:,np.newaxis,:,:],-1, (0.255,0), 1)
    # cv2.line(img_wall, (x2, y2), (x1,y1), (0,0,255), 2, cv2.LINE_AA)
    
    # cv2.line(img_wall, (x3, y3), (x5,y5), (0,255,0), 1, cv2.LINE_AA)
    # cv2.line(img_wall, (x0, y0), (x4,y4), (0,255,0), 1, cv2.LINE_AA)


    # # x1,x2,y1,y2=int(x1),int(x2),int(y1),int(y2)
    # l=((x1-x2)**2+(y1-y2)**2)**0.5
    # all_area=l*4
    # wall_line=[x1,y1,x2,y2]
    # # print(l,wall_line)
    
    



# cv2.imshow('gray',img_gray)
# cv2.imshow('houghp',img_houghp)
# cv2.imshow('fast LSD',img_fastLSD)
# cv2.imshow('LSD',img_lsd)
cv2.imshow('wall-line',img_wall)
cv2.imshow('K_map',K_map)
cv2.waitKey(0)
''''''