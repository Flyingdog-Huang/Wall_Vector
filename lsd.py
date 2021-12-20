# wall line 反向生长——改进为全局pridiction pix分类
# 斜率角度按照 +-90 - +-85 - +-75 - +-65 - +-55 - +-45 - +-35 - +-25 - +-15 - +-5 - 0



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
img_fastLSD=img_pre.copy()
img_fastLSD=lsd_dec.drawSegments(img_fastLSD,lines_fastLSD)

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

# save wall lines , direction and k
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
thr_dis=0.3 # diff condition

# K feature map
K_feature_map=np.zeros((y,x,3),np.uint8) # show K
K_map=np.zeros((y,x),np.float16) # save k
line_map=np.zeros((y,x),np.uint16) # save no(1,2,3...) of wall line, 0-not wall

# k max init
# k_max=0

# max/min K -- +-tan80
max_k=99999

# no of wall line
no_line=0

# 水平斜率阈值
k_max_horizontal=0.177
k_min_horizontal=-0.177

# 竖直斜率阈值
k_max_vertical=5.76
k_min_vertical=-5.76


# 定义斜率K的相似性
def is_K_same(k1,k2):
    # 夹角在10°内
    tan_a=(k1-k2)/(1+k1*k2+0.0001)
    if tan_a**2<=math.tan(10/180*math.pi):
        return True
    return False


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

    dx=0
    dy=a2
    if y1==y2:#(y1-y2)**2<=a1**2:
        # y1=int(y1)
        # y2=y1
        dx=0
        dy=a2
    # elif (x1-x2)**2<=a1**2:
    #     # x1=int(x1)
    #     x2=x1
    #     dx=a2
    #     dy=0
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
    mask_area=max(mask1_area,mask2_area)
    mask1_p=mask1_area/np.sum(mask1_b)
    mask2_p=mask2_area/np.sum(mask2_b)
    mask_p=max(mask1_p,mask2_p)

    if (mask1_p+mask2_p)>(thr_mp*2) and ((mask1_p-mask2_p)/mask_p)**2>(thr_dis**2) and mask_area>=20: # True:
        # num of wall line
        no_line+=1 

        # # draw wall lines
        # cv2.line(img_wall, (x2, y2), (x1,y1),(0,0,255), 1)

        # get k - wall line
        k_wallLine=None
        if x1==x2:
            k_wallLine=max_k
        elif y1==y2:
            k_wallLine=0
        else:
            k_wallLine=(line[0][1]-line[0][3])/(line[0][0]-line[0][2])  # (y1-y2)/(x1-x2)
            # k_2=k_2**2
        # print(k_2)

        # # 近似化处理斜率和坐标
        # if k_wallLine>k_max_vertical or k_wallLine<k_min_vertical:
        #     if (x1-x2)**2<=2**2:
        #         k_wallLine=max_k
        #         x1=int((x1+x2)/2)
        #         x2=x1

        # if k_wallLine<=k_max_horizontal and k_wallLine>=k_min_horizontal:
        #     if (y1-y2)**2<=2**2:
        #         k_wallLine=0 
        #         y1=int((y1+y2)/2)
        #         y2=y1

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
        # 此段wall line的轨迹
        line_flg=np.zeros((y,x),np.uint8)
        cv2.line(line_flg, (x2, y2), (x1,y1),1, 1)

        x_min=min(x1,x2)
        x_max=max(x1,x2)
        
        y_min=min(y1,y2)
        y_max=max(y1,y2)
        
        # line_map基于wall line轨迹，保存全局wall line的序号NO，与wall_line中保存的wall line坐标的序号一一对应
        # K_map 基于wall line轨迹，保存全局wall line的斜率K值，如果有交叉，以长度长的为准
        for y_line in range(y_min,y_max+1):
            for x_line in range(x_min,x_max+1):
                if line_flg[y_line,x_line]==1:
                    # 如果轨迹为空，则直接写入NO和K
                    if line_map[y_line,x_line]==0:
                        line_map[y_line,x_line]=no_line
                        # print('k',k)
                        K_map[y_line,x_line]=k_wallLine
                    # 若有交叉，以长度长的为准
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
        
        # print('line_map[y_min:y_max+1,x_min:x_max+1]',line_map[y_min:y_max+1,x_min:x_max+1])

        # add wall lines
        wall_line=[x1,y1,x2,y2]
        wall_lines.append(wall_line)
        # print('wall-line',line)

        # get direction
        direc_line=[x1,y1,x5,y5] if mask1_p>mask2_p else [x1,y1,x3,y3]
        direc_lines.append(direc_line)
        # print(direc_line)

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


#### prediction pix 分类阶段 ####

# save new no(1,2,3...) of wall line, 0-not wall
new_line_map=np.zeros((y,x),np.uint16) 

# save distance  from pix point to wall lines
dis_k_line_map=np.zeros((y,x),np.float16)

# 点距探测
def point_detect(x_point,y_point):
    r_detect=1
    is_detected=False
    while dis_k_line_map[y_point,x_point]==0:
        for y_detect_r in range(-r_detect,r_detect+1):
            y_detect=y_point+y_detect_r
            if y_detect<0:y_detect=0
            if y_detect>=y:y_detect=y-1
            for x_detect_r in range(-r_detect,r_detect+1):
                x_detect=x_point+x_detect_r
                if x_detect<0:x_detect=0
                if x_detect>=x:x_detect=x-1
                # 若探测到wall line 
                if line_map[y_detect,x_detect]>0:
                    dis_k_line_map[y_point,x_point]=r_detect # 赋值距离map
                    new_line_map[y_point,x_point]=line_map[y_detect,x_detect] # 赋值生长的new line map
                    K_map[y_point,x_point]=k_lines[line_map[y_detect,x_detect]-1] # 赋值生长的K map
                    is_detected=True
        
        # 若未探测到则增大探测半径继续探测
        if is_detected is False:
            r_detect+=1


# classify each prediction pix
for yi in range(y):
    for xi in range(x):
        # 若此点为prediction pix, 且不为wall line 轨迹
        if img_bool[yi,xi]==1 and line_map[yi,xi]==0:
            point_detect(xi,yi)
            

# print('finish classify each prediction pix')

#### k feature map充填阶段 ####

# wall line 方向的探测区域长度 
l_d=20 

# # save new k wall line(除了水平和竖直以外)
# k_new_wallLine=[]

# save the flag that pix is or not filled by same lines
is_fill_by_same_line=np.zeros((y,x),np.uint8) 

# create k map
for i in range(len(wall_lines)):
    x1,y1,x2,y2=wall_lines[i] # 提取wall line坐标
    k_line=k_lines[i] # 此条wall line的斜率K
    x1,y1,x_d,y_d=direc_lines[i] # 方向线坐标，以point1为起点

    # draw wall lines
    # 绘制原始wall line
    cv2.line(img_wall, (x2, y2), (x1,y1),(0,0,255), 1)
    # cv2.line(img_wall, (x2, y2), (x1,y1), (0,0,255), 2, cv2.LINE_AA)

    new_no=i+1 # 此时的wall line编号

    # draw line flag on line map
    # 绘制这条wall line 的轨迹（不水平和不竖直的wall line需要以此为生长种子）
    line_flg=np.zeros((y,x),np.uint8)
    cv2.line(line_flg, (x2, y2), (x1,y1),1, 1)

    # 分为：水平、垂直和其他情况分类处理
    # ROI区域
    y_min=min(y1,y2)
    y_max=max(y1,y2)
    
    x_min=min(x1,x2)
    x_max=max(x1,x2)
    

    # 垂直wall line -- x水平扩展
    if x1==x2: # k_line<k_min_vertical or k_line>k_max_vertical:
        # 获取X的方向
        k_direc=1 if x_d>x1 else -1
        for y_line in range(y_min,y_max+1):
            for x_line in range(x_min,x_max+1):
                # x=x_min+k_direc*ddx

                # 若存在wall line轨迹，则进行探测
                if line_flg[y_line,x_line]==1:
                    is_line=0 # 是否标记此点为wall line

                    # wall line 方向的探测阶段
                    for ddx in range(1,l_d+1):
                        x_k=x_line+k_direc*ddx  # 探测端点

                        if x_k<0 or x_k>=x: break # 超出img范围，则结束探测
                        # print('x_k',x_k)

                        # 如探测端点是直线或预测像素点，则启动充填
                        if line_map[y_line,x_k]>0 or img_bool[y_line,x_k]==1:
                            
                            # 充填阶段
                            for dddx in range(1,ddx+1):
                                dx_k=x_line+k_direc*dddx

                                # 当探测端点是wall line时
                                if line_map[y_line,x_k]>0:
                                    # 探测端点 wall line 信息
                                    k_detect=K_map[y_line,x_k] # 探测点wall line斜率

                                    # 如果端点直线K相似，直接充填
                                    if is_K_same(k_line,k_detect):
                                        is_line=1 # 此点保存                                     
                                        K_map[y_line,dx_k]=max_k # 赋值此点斜率值

                                        # 判断距离map数值
                                        # 若此点是被相似成对wall line生长过 —— 按最小值赋值
                                        if is_fill_by_same_line[y_line,dx_k]==1:
                                            dis_k_line_map[y_line,dx_k]=min(dddx,dis_k_line_map[y_line,dx_k])
                                        # 若此点未被相似成对wall line生长过 —— 直接赋值dis
                                        else:
                                            dis_k_line_map[y_line,dx_k]=dddx
                                        
                                        # 标记此点被相似成对线段充填
                                        is_fill_by_same_line[y_line,dx_k]=1
                                        is_fill_by_same_line[y_line,x_line]=1
                                        
                                        new_line_map[y_line,dx_k]=new_no # 赋值此点wall line序号
                                        continue
                                    
                                    # 如果端点直线不相似
                                    else:
                                        # 如果是未生长区域，直接充填
                                        if new_line_map[y_line,dx_k]==0  and line_map[y_line,dx_k]==0:
                                            is_line=1                                     
                                            K_map[y_line,dx_k]=max_k
                                            new_line_map[y_line,dx_k]=new_no
                                            dis_k_line_map[y_line,dx_k]=dddx
                                            continue

                                    #     # 如果是生长区域
                                    #     else:
                                    #         # 没有被相似成对直线情况充填时+距离最近时 = 可以充填
                                    #         if is_fill_by_same_line[y_line,dx_k]==0:
                                    #             if dis_k_line_map[y_line,dx_k]>0 and dis_k_line_map[y_line,dx_k]>dddx:
                                    #                 is_line=1                                     
                                    #                 K_map[y_line,dx_k]=max_k
                                    #                 new_line_map[y_line,dx_k]=new_no
                                    #                 dis_k_line_map[y_line,dx_k]=dddx
                                    #                 continue


                                # 如果是预测墙体区域 
                                # 如果是由对边线充填的形式，则不充填
                                if img_bool[y_line,x_k]==1:
                                    
                                    # 如果是未生长区域，直接充填
                                    if new_line_map[y_line,dx_k]==0 :
                                        is_line=1
                                        new_line_map[y_line,dx_k]=new_no                                     
                                        K_map[y_line,dx_k]=max_k
                                        dis_k_line_map[y_line,dx_k]=dddx
                                        continue
                                    
                                    # 如果是生长区域，没有被相似成对直线情况充填时+距离最近时 = 可以充填
                                    else:
                                        if is_fill_by_same_line[y_line,dx_k]==0:

                                            if dis_k_line_map[y_line,dx_k]>0 and dis_k_line_map[y_line,dx_k]>dddx:
                                                is_line=1                                     
                                                K_map[y_line,dx_k]=max_k
                                                new_line_map[y_line,dx_k]=new_no
                                                dis_k_line_map[y_line,dx_k]=dddx
                                                continue
                                
                    if is_line==1:
                        new_line_map[y_line,x_line]=new_no
                        dis_k_line_map[y_line,x_line]=0 
                        

    # 水平扩展
    elif y1==y2:
        k_direc=1 if y_d>y1 else -1
        for x_line in range(x_min,x_max+1):
            for y_line in range(y_min,y_max+1):

                # 若ROI区域有wall line轨迹
                if line_flg[y_line,x_line]==1:
                    is_line=0 # 是否标记此点为wall line

                    # 探测阶段
                    for ddy in range(1,l_d+1):
                        y_k=y_line+k_direc*ddy # 探测端点

                        if y_k<0 or y_k>=y: break # 若超出范围则停止探测

                        # 若探测端点是wall line 或是 pridiction mask
                        if line_map[y_k,x_line]>0 or img_bool[y_k,x_line]==1:

                            # 充填阶段
                            for dddy in range(1,ddy+1):
                                dy_k=y_line+k_direc*dddy

                                # 如果端点是wall line
                                if line_map[y_k,x_line]>0:
                                    # 探测端点 wall line 信息
                                    k_detect=K_map[y_k,x_line] # 探测点wall line斜率
                                    
                                    # 如果端点直线K相似，直接充填
                                    if is_K_same(k_line,k_detect):
                                        is_line=1   # 此点保存                      
                                        K_map[dy_k,x_line]=0  # 赋值此点斜率值

                                        # 判断距离map数值
                                        # 若此点是被相似成对wall line生长过 —— 按最小值赋值
                                        if is_fill_by_same_line[dy_k,x_line]==1:
                                            dis_k_line_map[dy_k,x_line]=min(dddy,dis_k_line_map[dy_k,x_line])
                                        # 若此点未被相似成对wall line生长过 —— 直接赋值dis
                                        else:
                                            dis_k_line_map[dy_k,x_line]=dddy
                                                                                
                                        # 标记此点被相似成对线段充填
                                        is_fill_by_same_line[dy_k,x_line]=1
                                        is_fill_by_same_line[y_line,x_line]=1
                                        # 赋值此点wall line序号
                                        new_line_map[dy_k,x_line]=new_no
                                        continue

                                    # 如果端点直线不相似
                                    else:
                                        #  如果是未生长区域，直接充填
                                        if new_line_map[dy_k,x_line]==0  and line_map[dy_k,x_line]==0:
                                            is_line=1                                     
                                            K_map[dy_k,x_line]=0
                                            new_line_map[dy_k,x_line]=new_no
                                            dis_k_line_map[dy_k,x_line]=dddy
                                            continue

                                    #     # 如果是生长区域，没有被相似成对直线情况充填时+dis更小=可以充填
                                    #     else:
                                    #         if is_fill_by_same_line[dy_k,x_line]==0:

                                    #             if dis_k_line_map[dy_k,x_line]>0 and dis_k_line_map[dy_k,x_line]>dddy:
                                                    
                                    #                 is_line=1                                     
                                    #                 K_map[dy_k,x_line]=0
                                    #                 new_line_map[dy_k,x_line]=new_no
                                    #                 dis_k_line_map[dy_k,x_line]=dddy
                                    #                 continue

                                # 如果是预测墙体区域
                                # 如果是由对边线充填的形式，则不充填
                                if img_bool[y_k,x_line]==1:

                                    # 如果是未生长区域，直接充填
                                    if new_line_map[dy_k,x_line]==0 :
                                        is_line=1
                                        new_line_map[dy_k,x_line]=new_no                                     
                                        K_map[dy_k,x_line]=0
                                        dis_k_line_map[dy_k,x_line]=dddy
                                        continue

                                    # 如果是生长区域，没有被相似成对直线情况充填时+dis更小=可以充填
                                    else:
                                        if is_fill_by_same_line[dy_k,x_line]==0:

                                            if dis_k_line_map[dy_k,x_line]>0 and dis_k_line_map[dy_k,x_line]>dddy:
                                                
                                                is_line=1
                                                new_line_map[dy_k,x_line]=new_no                                     
                                                K_map[dy_k,x_line]=0
                                                dis_k_line_map[dy_k,x_line]=dddy
                                                continue
                    
                    if is_line==1:
                        new_line_map[y_line,x_line]=new_no
                        dis_k_line_map[y_line,x_line]=0 
    
    # 斜边扩展
    else:
        xk_direc=1 if x_d>x1 else -1
        yk_direc=1 if y_d>y1 else -1
        dk=-1/k_line
        dx=(l_d**2/(1+dk**2))**0.5
        dy=dk*dx if dk>0 else -dk*dx
        # print(k_line,dk,dx)
        for x_line in range(x_min,x_max+1):
            for y_line in range(y_min,y_max+1):

                # 若ROI区域有wall line轨迹
                if line_flg[y_line,x_line]==1:

                    is_line=0 # 是否标记此点为wall line

                    # 确定过此点的直线方程
                    db=y_line-dk*x_line

                    # 确定探测端点
                    x_k=x_line+xk_direc*dx
                    y_k=dk*x_k+db
                    x_k,y_k=int(x_k),int(y_k)

                    # 绘制探测线段
                    detection_line=np.zeros((y,x),np.uint8)
                    cv2.line(detection_line, (x_line, y_line), (int(x_k),int(y_k)),1, 1)

                    # 探测阶段
                    # x轴探测
                    for ddx in range(1,int(dx)+1):
                        dx_k=x_line+xk_direc*ddx
                        dy_k=dk*dx_k+db
                        dx_k=int(dx_k)
                        dy_k=int(dy_k)

                        # 超出img范围，则结束探测
                        if dx_k<0 or dx_k>=x or dy_k<0 or dy_k>=y:break
                        # is_fill=0

                        # 确定y轴探测范围（y_line，dy_k）
                        detec_y_min=min(y_line,dy_k)
                        if detec_y_min<0:detec_y_min=0
                        detec_y_max=max(y_line,dy_k)
                        if detec_y_max>y-1:detec_y_max=y-1

                        # y轴探测
                        for detec_y in range(detec_y_min,detec_y_max+1):

                            # 当是探测线时，检查是否符合填充条件
                            if detection_line[detec_y,int(dx_k)]==1 :
                                
                                # 如探测端点是直线或预测像素点，则启动充填
                                if line_map[detec_y,int(dx_k)]>0 or img_bool[detec_y,int(dx_k)]==1:

                                    # ROI区域
                                    de_x_min=int(min(x_line,dx_k))
                                    if de_x_min<0:de_x_min=0
                                    de_x_max=int(max(x_line,dx_k))
                                    if de_x_max>=x:de_x_max=x-1

                                    de_y_min=int(min(y_line,dy_k))
                                    if de_y_min<0:de_y_min=0
                                    de_y_max=int(max(y_line,dy_k))
                                    if de_y_max>=y:de_y_max=y-1
                                    # print(x)
                                    # print(x_line,dx_k)
                                    # print(de_x_min,de_x_max)

                                    for de_x in range(de_x_min,de_x_max+1):
                                        for de_y in range(de_y_min,de_y_max+1):
                                            # print(de_y,de_x)

                                            # 探测线存在且不为wall line上的点
                                            if detection_line[de_y,de_x]==1 :
                                                if de_y!=y_line or de_x!=x_line:

                                                    # 计算距离值
                                                    distance_point=((de_y-y_line)**2+(de_x-x_line)**2)**0.5

                                                    k_now=k_line
                                                    # tan_k_p10_down=1-k_now*0.176
                                                    # tan_k_n10_down=1+k_now*0.176
                                                    # tan_k_p10=None
                                                    # tan_k_n10=None
                                                    # if tan_k_p10_down==0:
                                                    #     tan_k_p10=6
                                                    # else:
                                                    #     tan_k_p10=(k_now+0.176)/tan_k_p10_down
                                                    # if tan_k_n10_down==0:
                                                    #     tan_k_n10=6
                                                    # else:
                                                    #     tan_k_n10=(k_now-0.176)/tan_k_n10_down
                                                    
                                                    # 如果探测区域有直线存在
                                                    if line_map[detec_y,int(dx_k)]>0 :

                                                        # 获取探测端点 wall line 信息
                                                        k_detect=K_map[detec_y,int(dx_k)] # 获取斜率K
                                                        
                                                        # 且斜率K近似, 可以直接充填
                                                        if is_K_same(k_line,k_detect):
                                                            is_line=1  # 此点保存  
                                                            K_map[de_y,de_x]=k_now # 赋值此点斜率值

                                                            # 判断距离map数值
                                                            # 若此点是被相似成对wall line生长过 —— 按最小值赋值
                                                            if is_fill_by_same_line[de_y,de_x]==1:
                                                                dis_k_line_map[de_y,de_x]=min(distance_point,dis_k_line_map[de_y,de_x])
                                                            # 若此点未被相似成对wall line生长过 —— 直接赋值dis
                                                            else:
                                                                dis_k_line_map[de_y,de_x]=distance_point
                                                            
                                                            # 标记此点被相似成对线段充填
                                                            is_fill_by_same_line[de_y,de_x]=1
                                                            is_fill_by_same_line[y_line,x_line]=1
                                                            
                                                            # 赋值此点wall line序号
                                                            new_line_map[de_y,de_x]=new_no
                                                            continue

                                                        # 如果端点直线不相似
                                                        else:
                                                            # 如果是未生长区域 + 此点不为直线，直接充填
                                                            if new_line_map[de_y,de_x]==0 and line_map[de_y,de_x]==0:
                                                                is_line=1 # 此点保存 
                                                                K_map[de_y,de_x]=k_now
                                                                new_line_map[de_y,de_x]=new_no
                                                                dis_k_line_map[de_y,de_x]=distance_point
                                                                continue
                                                            
                                                        #     # 如果是生长区域
                                                        #     else:
                                                        #         # 没有被相似成对直线情况充填时+距离最近时 = 可以充填
                                                        #         if is_fill_by_same_line[de_y,de_x]==0:

                                                        #             if dis_k_line_map[de_y,de_x]>0 and dis_k_line_map[de_y,de_x]>distance_point:
                                                        #                 is_line=1 # 此点保存 
                                                        #                 K_map[de_y,de_x]=k_now
                                                        #                 new_line_map[de_y,de_x]=new_no
                                                        #                 dis_k_line_map[de_y,de_x]=distance_point
                                                        #                 continue

                                                    # 如果探测区域有墙体预测区域存在
                                                    if img_bool[detec_y,int(dx_k)]==1 :

                                                        # 如果是未生长区域，直接充填
                                                        if new_line_map[de_y,de_x]==0:
                                                            is_line=1
                                                            K_map[de_y,de_x]=k_now
                                                            new_line_map[de_y,de_x]=new_no
                                                            dis_k_line_map[de_y,de_x]=distance_point
                                                            continue

                                                        # 如果是生长区域，没有被相似成对直线情况充填时 + dis更小=可以充填
                                                        else:
                                                            if is_fill_by_same_line[de_y,de_x]==0:
                                                                
                                                                if dis_k_line_map[de_y,de_x]>0 and dis_k_line_map[de_y,de_x]>distance_point:
                                                                    is_line=1    
                                                                    K_map[de_y,de_x]=k_now
                                                                    new_line_map[de_y,de_x]=new_no
                                                                    dis_k_line_map[de_y,de_x]=distance_point
                                                                    continue

                    if is_line==1:
                        new_line_map[y_line,x_line]=new_no
                        dis_k_line_map[y_line,x_line]=0 


# show k feature map
color_b=[255,0,0] # K:-6_-2
color_g=[0,255,0] # K:-2_2
color_r=[0,0,255] # K:2_6
color_w=[255,255,255] # 显示水平和竖直

for i in range(x):
    for j in range(y):
        color=[0,0,0]
        if new_line_map[j][i]>0:
            if K_map[j][i]==max_k or K_map[j][i]==0: #(K_map[j][i]>=5.67 or K_map[j][i]<=-5.67) or (K_map[j][i]>=-0.176 and K_map[j][i]<=0.176):
                color=color_w
            elif K_map[j][i]<-2:
                color=color_b
            elif K_map[j][i]>=-2 and K_map[j][i]<=2:
                color=color_g
            else:
                color=color_r
        K_feature_map[j,i,:]=color




######### 区域生长阶段 ############


def isPointIn(point_x,point_y,x_b_min=0,x_b_max=x,y_b_min=0,y_b_max=y):
    #判断点是否在某一区域内
    if point_x>=x_b_min and point_x<x_b_max and point_y>=y_b_min and point_y<y_b_max:
        return True
    else:
        return False



# K list去重
new_k_list=list(set(k_lines))


def k_RG(point_x,point_y,k_stand,k_rg_map,k_rg_flag_map,k_size=3,thr=0.5):
    # 区域生长
    # 8邻域
    # 0.5
    # 判断是否已经遍历过
    if k_rg_flag_map[point_y,point_x]!=255:
        k_rg_flag_map[point_y,point_x]=255 # 标记已经遍历
        
        x_nears=[]
        y_nears=[]
        for i in range(k_size):
            for j in range(k_size):
                x_near=point_x+i-1
                y_near=point_y+j-1
                if isPointIn(x_near,y_near):
                    x_nears.append(x_near)
                    y_nears.append(y_near)
        
        num_near=len(x_nears)
        print(x_nears,y_nears)
        # 是否属于K区域
        num_k=0
        num_same_k=0
        for i in range(num_near):
            x_near,y_near=x_nears[i],y_nears[i]
            if new_line_map[y_near,x_near]>0 :
                num_k+=1
                if is_K_same(k_stand,K_map[y_near,x_near]): num_same_k+=1
                
        is_k_area=False
        if num_same_k>=int(num_k*thr):
            is_k_area=True
            k_rg_map[point_y,point_x]=255
                
        
        # 四周生长
        if is_k_area:
            for i in range(num_near):
                x_near,y_near=x_nears[i],y_nears[i]
                if k_rg_flag_map[y_near,x_near]!=255:
                    k_RG(x_near,y_near,k_stand=k_stand,k_rg_map=k_rg_map,k_rg_flag_map=k_rg_flag_map,k_size=k_size,thr=thr)
       


# 首先将垂直和水平的区域分割出来
k_rg=0
rg_map=np.zeros((y,x),np.uint8)
rg_flag_map=np.zeros((y,x),np.uint8)

for x_rg in range(x):
    for y_rg in range(y):
        if new_line_map[y_rg,x_rg]>0 and is_K_same(K_map[y_rg,x_rg],k_rg) :
            k_RG(x_rg,y_rg,k_stand=k_rg,k_rg_map=rg_map,k_rg_flag_map=rg_flag_map)

k_rg=max_k
rg_map1=np.zeros((y,x),np.uint8)
rg_flag_map1=np.zeros((y,x),np.uint8)

for x_rg in range(x):
    for y_rg in range(y):
        if new_line_map[y_rg,x_rg]>0 and is_K_same(K_map[y_rg,x_rg],k_rg) :
            k_RG(x_rg,y_rg,k_stand=k_rg,k_rg_map=rg_map1,k_rg_flag_map=rg_flag_map1)


# 基于斜率K的坐标正-反变换
def pointChange(x_point,y_point,k_rec,is_toK=True):
    cos_a=(1/k_rec**2)**0.5
    sin_a=k_rec*cos_a

    if is_toK is False:
        sin_a=-sin_a
    new_x_point=x_point*cos_a+y_point*sin_a
    new_y_point=y_point*cos_a-x_point*sin_a
    return int(new_x_point),int(new_y_point)


# 基于斜率K的方向矩形拟合优化
def findKminRect(k_rec,cnt):
    bbox_points=[] # [point1,point2,point3,point4]

    if k_rec==0 or k_rec==6:
        x_rec,y_rec,w_rec,h_rec=cv2.boundingRect(cnt)
        x1,y1=x_rec,y_rec
        x2,y2=x_rec+w_rec,y_rec
        x3,y3=x_rec+w_rec,y_rec+h_rec
        x4,y4=x_rec,y_rec+h_rec
    else:
        # 坐标转换
        new_cnt=[]

        for point in cnt:
            new_point=[]
            x_point,y_point=point[0]
            new_x_point,new_y_point=pointChange(x_point,y_point,k_rec)

            new_point=[new_x_point,new_y_point]
            new_cnt.append(new_point)

        new_cnt=np.array(new_cnt)
        new_cnt=new_cnt[:,np.newaxis,:]

        # 在K坐标系下拟合rec
        x_rec,y_rec,w_rec,h_rec=cv2.boundingRect(cnt)
        x1,y1=x_rec,y_rec
        x2,y2=x_rec+w_rec,y_rec
        x3,y3=x_rec+w_rec,y_rec+h_rec
        x4,y4=x_rec,y_rec+h_rec

        # 转换回原始坐标系
        x1,y1=pointChange(x1,y1,k_rec,is_toK=False)
        x2,y2=pointChange(x2,y2,k_rec,is_toK=False)
        x3,y3=pointChange(x3,y3,k_rec,is_toK=False)
        x4,y4=pointChange(x4,y4,k_rec,is_toK=False)



    bbox_points=[x1,y1,x2,y2,x3,y3,x4,y4]
    return bbox_points



vector_wall=img_pre.copy()
new_wall_mask=np.zeros((y,x),np.uint8)
total_num_wall=0

# 拟合轮廓成矩形
contours, _ = cv2.findContours(rg_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours1, _ = cv2.findContours(rg_map1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# ***基于:轮廓面积>area_thr---过滤杂点
for cnt in contours:
    rect= cv2.minAreaRect(cnt)
    points=cv2.boxPoints(rect)
    points=np.int0(points)
    # print('points',points)
    l1=max(int(((points[0][0]-points[1][0])**2)**0.5),int(((points[0][1]-points[1][1])**2)**0.5))
    l2=max(int(((points[2][0]-points[1][0])**2)**0.5),int(((points[2][1]-points[1][1])**2)**0.5))
    width_wall=min(l1,l2)
    x_e_vector=(points[0][0]+points[1][0])/2 if width_wall==l1 else (points[2][0]+points[1][0])/2
    y_e_vector=(points[0][1]+points[1][1])/2 if width_wall==l1 else (points[2][1]+points[1][1])/2
    x_s_vector=(points[2][0]+points[3][0])/2 if width_wall==l1 else (points[0][0]+points[3][0])/2
    y_s_vector=(points[2][1]+points[3][1])/2 if width_wall==l1 else (points[0][1]+points[3][1])/2

    cv2.line(vector_wall, (int(x_e_vector), int(y_e_vector)), (int(x_s_vector),int(y_s_vector)),(0,0,255), 1)
    total_num_wall+=1
    print('{0} - wall line: [[x_e_vector,y_e_vector],[x_s_vector,y_s_vector],width_wall]={1}'.format(total_num_wall,[[x_e_vector,y_e_vector],[x_s_vector,y_s_vector],width_wall]))
    print('points',points)
    cv2.drawContours(new_wall_mask,[points],0,255,-1)

for cnt in contours1:
    rect= cv2.minAreaRect(cnt)
    points=cv2.boxPoints(rect)
    points=np.int0(points)
    l1=max(int(((points[0][0]-points[1][0])**2)**0.5),int(((points[0][1]-points[1][1])**2)**0.5))
    l2=max(int(((points[2][0]-points[1][0])**2)**0.5),int(((points[2][1]-points[1][1])**2)**0.5))
    width_wall=min(l1,l2)
    x_e_vector=(points[0][0]+points[1][0])/2 if width_wall==l1 else (points[2][0]+points[1][0])/2
    y_e_vector=(points[0][1]+points[1][1])/2 if width_wall==l1 else (points[2][1]+points[1][1])/2
    x_s_vector=(points[2][0]+points[3][0])/2 if width_wall==l1 else (points[0][0]+points[3][0])/2
    y_s_vector=(points[2][1]+points[3][1])/2 if width_wall==l1 else (points[0][1]+points[3][1])/2

    cv2.line(vector_wall, (int(x_e_vector), int(y_e_vector)), (int(x_s_vector),int(y_s_vector)),(0,0,255), 1)
    total_num_wall+=1
    print('{0} - wall line: [[x_e_vector,y_e_vector],[x_s_vector,y_s_vector],width_wall]={1}'.format(total_num_wall,[[x_e_vector,y_e_vector],[x_s_vector,y_s_vector],width_wall]))
    print('points',points)
    cv2.drawContours(new_wall_mask,[points],0,255,-1)

'''
other_k_map=np.zeros((y,x),np.uint8)

# 其余K值
for k_rg in new_k_list:
    # k_rg=6
    rg_map=np.zeros((y,x),np.uint8)
    rg_flag_map=np.zeros((y,x),np.uint8)

    for x_rg in range(x):
        for y_rg in range(y):
            if new_line_map[y_rg,x_rg]>0 and is_K_same(K_map[y_rg,x_rg],k_rg):
                k_RG(x_rg,y_rg,k_stand=k_rg,k_rg_map=rg_map,k_rg_flag_map=rg_flag_map)

    other_k_map+=rg_map
    # 拟合轮廓成矩形
    contours, _ = cv2.findContours(rg_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x1,y1,x2,y2,x3,y3,x4,y4=findKminRect(k_rg,cnt)
        l1=((x1-x2)**2+(y1-y2)**2)**0.5
        l2=((x3-x2)**2+(y3-y2)**2)**0.5
        width_wall=min(l1,l2)
        x_e_vector=(x1+x2)/2 if width_wall==l1 else (x3+x2)/2
        y_e_vector=(y1+y2)/2 if width_wall==l1 else (y3+y2)/2
        x_s_vector=(x3+x4)/2 if width_wall==l1 else (x1+x4)/2
        y_s_vector=(y3+y4)/2 if width_wall==l1 else (x3+x4)/2

        cv2.line(vector_wall, (int(x_e_vector), int(y_e_vector)), (int(x_s_vector),int(y_s_vector)),(0,0,255), 1)
        total_num_wall+=1
        print('{0} - wall line: [[x_e_vector,y_e_vector],[x_s_vector,y_s_vector],width_wall]={1}'.format(total_num_wall,[[x_e_vector,y_e_vector],[x_s_vector,y_s_vector],width_wall]))
        points=[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        points=np.array(points)
        print('points',points)
        cv2.drawContours(new_wall_mask,[points],0,255,-1)

''' 
'''
# cv2.imshow('gray',img_gray)
# cv2.imshow('houghp',img_houghp)
# cv2.imshow('LSD',img_lsd)
# cv2.imshow('img_bool',img_bool)
cv2.imshow('fast LSD',img_fastLSD)
cv2.imshow('rg_map1',rg_map1)
cv2.imshow('rg_map',rg_map)
cv2.imshow('other_k_map',other_k_map)
'''

cv2.imshow('K_feature_map',K_feature_map)

cv2.imshow('new_wall_mask',new_wall_mask)
cv2.imshow('vector_wall',vector_wall)
cv2.imshow('wall-line',img_wall)

cv2.waitKey(0)
