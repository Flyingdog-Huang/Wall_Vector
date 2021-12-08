import cv2
import numpy as np
from pylsd.lsd import lsd
import math

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
# print(img_bool.shape)
_,img_bool=cv2.threshold(img_bool,127,1,cv2.THRESH_BINARY)#二值化
# print(img_bool.shape)

# process bool img : 膨胀=>腐蚀(闭运算)——充填空隙
# k_dila=3
# k_ero=3
# it_dila=1
# it_ero=1
# # 定义核结构
# kernel_dila= np.ones((k_dila,k_dila),np.uint8) # cv2.getStructuringElement(cv2.MORPH_RECT,(k_dila,k_dila))
# kernel_ero= np.ones((k_ero,k_ero),np.uint8) # cv2.getStructuringElement(cv2.MORPH_RECT,(k_ero,k_ero))
# # 膨胀
# img_bool=cv2.dilate(img_bool,kernel_dila,it_dila)
# # 腐蚀
# img_bool=cv2.erode(img_bool,kernel_ero,it_ero)
# close opration
# k=3
# it=1
# kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
# img_bool=cv2.morphologyEx(img_bool, cv2.MORPH_CLOSE, kernel, iterations=it)

# img_bool=cv2.cvtColor(img_bool, cv2.COLOR_BGR2GRAY)#灰化
# _,img_bool=cv2.threshold(img_bool,127,1,cv2.THRESH_BINARY)#二值化
# print(img_bool)
# print(img_bool.shape)

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
thr_mp=0.1 # exist condition
thr_dis=0.4 # diff condition

# K feature map
K_feature_map=np.zeros((y,x,3),np.uint8) # show K
K_map=np.zeros((y,x),np.uint8) # save k
line_map=np.zeros((y,x),np.uint8) # save no(1,2,3...) of wall line, 0-not wall

# k max init
# k_max=0

# max/min K -- +-tan80
max_k=6 # tan80
min_k=-6 # -tan80

# no of wall line
no_line=0

# filter wall lines
for line in lines_fastLSD:
    # print(line[0])
    x1,y1,x2,y2=line[0]
    if (x1-x2)**2<=a1**2 and (y1-y2)**2<=a1**2:
        continue
    # print('(x1-x2)**2',(x1-x2)**2)
    # print('a1**2',a1**2)
    # print('(y1-y2)**2',(y1-y2)**2)
    # print('a1**2',a1**2)
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
    mask_p=max(mask1_p,mask2_p)

    if (mask1_p+mask2_p)>(thr_mp*2) and ((mask1_p-mask2_p)/mask_p)**2>(thr_dis**2): # True:
        # add wall lines
        no_line+=1 # add no of wall line
        wall_line=[x1,y1,x2,y2]
        wall_lines.append(wall_line)
        # print('wall-line',line)

        # get direction
        direc_line=[x1,y1,x5,y5] if mask1_p>mask2_p else [x1,y1,x3,y3]
        direc_lines.append(direc_line)
        # print(direc_line)

        # get k
        k=max_k
        if x1==x2:
            k=max_k
        elif y1==y2:
            k=0
        else:
            k=(line[0][1]-line[0][3])/(line[0][0]-line[0][2])  # (y1-y2)/(x1-x2)
            # k_2=k_2**2
        # print(k_2)
        if k>max_k:
            k=max_k
        if k<min_k: # tan100
            k=min_k
        # if k_2<999999:
        #     k_max=max(k_2,k_max)
        k_lines.append(k)
        # print('k',k)

        # bool wall direction
        # t1=1 if mask2_p>mask1_p else -1
        # t2=1 if mask1_p>mask2_p else -1
        
        # k color
        a_k_color=int(255*3*(k-min_k)/(max_k-min_k))
        # print('K color',a_k_color)
        k_b,k_g,k_r=0,0,0
        i_color=a_k_color//255
        num_color=a_k_color%255
        if i_color==0:
            k_b=255
            k_g=num_color
        if i_color==1:
            k_g=255
            k_r=num_color
        if i_color==2:
            k_r=255
            k_b=num_color
        if i_color==3:
            k_b,k_g,k_r=255,255,255
        k_color=(k_b,k_g,k_r)
        # print('k_color',k_color)

        # draw K color line on k feature map
        cv2.line(K_feature_map, (x2, y2), (x1,y1),k_color, 1)

        # draw k on k map
        # cv2.line(K_map, (x2, y2), (x1,y1),int(k), 1)

        # draw line flag on line map
        line_flg=np.zeros((y,x),np.uint8)
        cv2.line(line_flg, (x2, y2), (x1,y1),1, 1)

        x_min=x1
        x_max=x2
        if x1>x2:
            x_min=x2
            x_max=x1
        y_min=y1
        y_max=y2
        if y1>y2:
            y_min=y2
            y_max=y1
        for y_line in range(y_min,y_max+1):
            for x_line in range(x_min,x_max+1):
                if line_flg[y_line,x_line]==1:
                    if K_map[y_line,x_line]==0:
                        K_map[y_line,x_line]=no_line
                    else:
                        # keep longer line k
                        l_now=(x1-x2)**2+(y1-y2)**2
                        no_befor=K_map[y_line,x_line]-1
                        print(no_befor)
                        l_befor=(wall_lines[no_befor][0]-wall_lines[no_befor][2])**2+(wall_lines[no_befor][1]-wall_lines[no_befor][3])**2
                        if l_now>l_befor:
                            K_map[y_line,x_line]=no_line
        # print('line_map:',line_map[y_min:y_max+1,x_min:x_max+1])
        # print('k map-B:',K_map[y_min:y_max+1,x_min:x_max+1,0])
        # print('k map-G:',K_map[y_min:y_max+1,x_min:x_max+1,1])
        # print('k map-R:',K_map[y_min:y_max+1,x_min:x_max+1,2])
        # print('K_map:',K_map[y_min:y_max+1,x_min:x_max+1])

        # # draw k feature map
        # if t1==-1:
        #     cv2.drawContours(K_map,rec1[np.newaxis,:,:,:],0, k_color, t1)
        # else:
        #     cv2.drawContours(K_map,rec2[np.newaxis,:,:,:],0, k_color, t2)


        # draw wall direction area
        # cv2.drawContours(img_wall,rec1[np.newaxis,:,:,:],0, (0,255,0), t1)
        # cv2.drawContours(img_wall,rec2[np.newaxis,:,:,:],0, (0,255,0), t2)

        # draw line
        # cv2.line(img_wall, (x2, y2), (x1,y1), (0,0,255), 2, cv2.LINE_AA)
        
        # print('-------------------')


    
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

# test wall line msg
# print(len(wall_lines))
# print(len(direc_lines))
# print(len(k_lines))

'''   
# 探测区域长度
l_d=20
# create k**2 map
for i in range(len(wall_lines)):
    x1,y1,x2,y2=wall_lines[i]
    k_line=k_lines[i]
    x1,y1,x_d,y_d=direc_lines[i]
    # cv2.line(img_wall, (x2, y2), (x1,y1), (0,0,255), 2, cv2.LINE_AA)

    # 分为：水平、垂直和其他情况分类处理
    y_min=y1
    y_max=y2
    if y1>y2:
        y_min=y2
        y_max=y1
    # dy=y_max-y_min+1
    x_min=x1
    x_max=x2
    if x1>x2:
        x_min=x2
        x_max=x1
    # dx=x_max-x_min+1

    # 垂直
    if x1==x2:
        for y in range(y_min,y_max+1):
            k_direc=1 if x_d>x1 else -1
            for ddx in range(l_d):
                x=x_min+k_direc*ddx
                if line_map[y,x]==1 and 



    # 水平
    elif y1==y2:
        pass
    else:
        pass




# cv2.imshow('gray',img_gray)
# cv2.imshow('houghp',img_houghp)
# cv2.imshow('fast LSD',img_fastLSD)
# cv2.imshow('LSD',img_lsd)
# cv2.imshow('img_bool',img_bool)
# cv2.imshow('wall-line',img_wall)

cv2.imshow('K_feature_map',K_feature_map)
cv2.waitKey(0)
'''