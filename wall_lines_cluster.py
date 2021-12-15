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

# Fast LSD
lsd_dec=cv2.ximgproc.createFastLineDetector()
lines_fastLSD=lsd_dec.detect(img_gray)

# filter lines
img_wall=img_pre.copy()

# get BINARY prediction img
img_bool=img_pre.copy()
img_bool=cv2.cvtColor(img_bool, cv2.COLOR_BGR2GRAY)#灰化
# print(img_bool.shape)
_,img_bool=cv2.threshold(img_bool,127,1,cv2.THRESH_BINARY)#二值化
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
K_map=np.zeros((y,x),np.float16) # save k
line_map=np.zeros((y,x),np.uint16) # save no(1,2,3...) of wall line, 0-not wall

# k max init
# k_max=0

# max/min K -- +-tan80
max_k=5.67 # tan80
min_k=-5.67 # -tan80

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
        # num of wall line
        no_line+=1 

        # # draw wall lines
        # cv2.line(img_wall, (x2, y2), (x1,y1),(0,0,255), 1)

        # get k - wall line
        k_wallLine=None
        if x1==x2:
            k_wallLine=6
        elif y1==y2:
            k_wallLine=0
        else:
            k_wallLine=(line[0][1]-line[0][3])/(line[0][0]-line[0][2])  # (y1-y2)/(x1-x2)
            # k_2=k_2**2
        # print(k_2)
        if k_wallLine>max_k or k_wallLine<min_k:
            k_wallLine=6
            x1=int((x1+x2)/2)
            x2=x1

        if k_wallLine<=0.176 and k_wallLine>=-0.176:
            k_wallLine=0 
            y1=int((y1+y2)/2)
            y2=y1

        k_lines.append(k_wallLine)
        # print('k',k)

        # bool wall direction
        t1=1 if mask2_p>mask1_p else -1
        t2=1 if mask1_p>mask2_p else -1
        
        # k color
        # a_k_color=int(255*3*(k-min_k)/(max_k-min_k))
        # # print('K color',a_k_color)
        # k_b,k_g,k_r=0,0,0
        # i_color=a_k_color//255
        # num_color=a_k_color%255
        # if i_color==0:
        #     k_b=255
        #     k_g=num_color
        # if i_color==1:
        #     k_g=255
        #     k_r=num_color
        # if i_color==2:
        #     k_r=255
        #     k_b=num_color
        # if i_color==3:
        #     k_b,k_g,k_r=255,255,255
        # k_color=(k_b,k_g,k_r)
        # # print('k_color',k_color)

        # # draw K color line on k feature map
        # cv2.line(K_feature_map, (x2, y2), (x1,y1),k_color, 1)

        # draw k on k map
        # cv2.line(K_map, (x2, y2), (x1,y1),int(k), 1)

        # draw line flag on line map
        line_flg=np.zeros((y,x),np.uint8)
        cv2.line(line_flg, (x2, y2), (x1,y1),1, 1)

        x_min=min(x1,x2)
        x_max=max(x1,x2)
        
        y_min=min(y1,y2)
        y_max=max(y1,y2)
        
        for y_line in range(y_min,y_max+1):
            for x_line in range(x_min,x_max+1):
                if line_flg[y_line,x_line]==1:
                    if line_map[y_line,x_line]==0:
                        line_map[y_line,x_line]=no_line
                        # print('k',k)
                        K_map[y_line,x_line]=k_wallLine
                    else:
                        # keep longer line k
                        l_now=(x1-x2)**2+(y1-y2)**2
                        no_befor=line_map[y_line,x_line]-1
                        # print(no_befor)
                        l_befor=(wall_lines[no_befor][0]-wall_lines[no_befor][2])**2+(wall_lines[no_befor][1]-wall_lines[no_befor][3])**2
                        if l_now>l_befor:
                            line_map[y_line,x_line]=no_line
                            # print('k',k)
                            K_map[y_line,x_line]=k_wallLine
                        
                        # print('len(k_lines)',len(k_lines))
                        # print('line_map[y_line,x_line]',line_map[y_line,x_line])
                        # K_map[y_line,x_line]=k_lines[line_map[y_line,x_line]-1]
        
        
        # add wall lines
        wall_line=[x1,y1,x2,y2]
        wall_lines.append(wall_line)
        # print('wall-line',line)

        # get direction
        direc_line=[x1,y1,x5,y5] if mask1_p>mask2_p else [x1,y1,x3,y3]
        direc_lines.append(direc_line)
        # print(direc_line)

# 探测区域长度 
l_d=20 
# save new no(1,2,3...) of wall line, 0-not wall
new_line_map=np.zeros((y,x),np.uint16) 
# save new k wall line
k_new_wallLine=[]
# 
new_wall_lines=[]

# change the lengh of wall line 
for i in range(len(wall_lines)):
    x1,y1,x2,y2=wall_lines[i]
    k_line=k_lines[i]
    x1,y1,x_d,y_d=direc_lines[i]

    # draw wall lines
    cv2.line(img_wall, (x2, y2), (x1,y1),(0,0,255), 1)

    new_no=i+1

    # draw line flag on line map
    line_flg=np.zeros((y,x),np.uint8)
    cv2.line(line_flg, (x2, y2), (x1,y1),1, 1)

    # 分为：水平、垂直和其他情况分类处理
    # ROI区域
    y_min=min(y1,y2)
    y_max=max(y1,y2)
    
    x_min=min(x1,x2)
    x_max=max(x1,x2)

    # 垂直修正
    if x1==x2:
        x_line=x1
        k_direc=1 if x_d>x1 else -1
        y_is_min=False
        while y_is_min==False:
            # 探测阶段
            for ddx in range(1,l_d+1):
                x_k=x_line+k_direc*ddx
                if x_k<0 or x_k>=x: break
                if line_map[y_min,x_k]>0 or img_bool[y_min,x_k]==1:
                    y_is_min=True
                    break
            if y_is_min==False:


cv2.imshow('wall-line',img_wall)
cv2.waitKey(0)
