import glob
import math

import cv2
import numpy as np

# 棋盘格设置
chessboard_width = 7
chessboard_height = 10
chessboard_length = 5 # mm

# 查看当前路径
# import os
# print(os.getcwd())

# 读取图片
scale = 0.5
date = 'mindvision'
suffix = 'jpg'
images_filenames = glob.glob(f'./calibration/images/{date}/*.{suffix}')

objpoints = [] # 在世界坐标系中的三维点
imgpoints = [] # 在图像平面的二维点
# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((chessboard_width*chessboard_height,3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_width,0:chessboard_height].T.reshape(-1,2)
objp = objp*chessboard_length

i = 0
shape = 0
for filename in images_filenames:
    # print(filename)
    img = cv2.imread(filename)
    h, w = img.shape[0], img.shape[1]

    # 缩放
    # img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    img_copy = img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = gray_img.shape

    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray_img, (chessboard_width, chessboard_height), None)

    if ret:
        print('i:', i)
        i += 1
        # (11, 11) 搜索窗口大小
        cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)

        j = 0
        start_corner = None
        end_corner = None
        for corner in corners:
            # 绘制角点编号
            pos = [int(corner[0][0]), int(corner[0][1])]
            
            cv2.putText(img_copy, str(j), pos, cv2.FONT_HERSHEY_SIMPLEX, 
                        int(3*scale), (0, 0, 255), 2)
            # 统计是否到下一行
            m = j % chessboard_width
            if m == 0:
                cv2.circle(img_copy, pos, 5, (0, 255, 0), -1)
                start_corner = corner
            elif m == chessboard_width-1:
                cv2.circle(img_copy, pos, 5, (0, 255, 0), -1)
                # 计算距离
                end_corner = corner
                pixel_dis = math.sqrt((end_corner[0][0] - start_corner[0][0]) ** 2
                                       + (end_corner[0][1] - start_corner[0][1]) ** 2)
                print('row pixel distance:', pixel_dis)
            j = j + 1

        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (chessboard_width, chessboard_height), corners, ret)
        cv2.imshow('findCorners', img)
        cv2.imshow('img', img_copy)
        cv2.waitKey(0)

# 标定
if i == 0:
    raise RuntimeError('未检测到有效棋盘格角点，无法标定。')

# calibrateCamera 的 imageSize 需要 (width, height)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (shape[1], shape[0]), None, None)

if not ret:
    raise RuntimeError('相机标定失败。')

# 计算像素到物理距离比例（先去畸变，再统计每行首尾角点距离）
row_pixel_dist_sum = 0.0
row_count = 0
for corners in imgpoints:
    undist_corners = cv2.undistortPoints(corners, mtx, dist, P=mtx)
    for row in range(chessboard_height):
        idx0 = row * chessboard_width
        idx1 = idx0 + chessboard_width - 1
        p0 = undist_corners[idx0][0]
        p1 = undist_corners[idx1][0]
        row_pixel_dist_sum += np.linalg.norm(p1 - p0)
        row_count += 1

if row_count == 0:
    raise RuntimeError('未统计到有效角点距离，无法计算比例。')

dis_avg = row_pixel_dist_sum / row_count

# proportion: 原始图像的 mm/px（当前 corners 在缩放图上检测得到）
proportion = chessboard_length * (chessboard_width - 1) / dis_avg
# proportion_scaled: 缩放后图像的 mm/px
proportion_scaled = proportion / scale


print('valid images:', i)
print('dis_avg(px, scaled image):', dis_avg)
print('proportion_scaled(mm/px, scaled image):', proportion_scaled)
print('proportion(mm/px, original image):', proportion)

# dis1 = 1.163
# dis2 = (0.903+0.893+0.893+0.913)/4
# proportion2 = proportion*dis1/dis2

# print("proportion2:", proportion2)

print("mtx:\n",mtx)      # 内参数矩阵
print("dist畸变值:\n",dist   )   # 畸变系数 distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
# print(shape)
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, shape[:2], 0, shape[:2])

# for filename in images_filenames:
#     img = cv2.imread(filename)
#     h, w = img.shape[0], img.shape[1]
#     # 缩放
#     if scale != 1:
#         img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
#     h, w = img.shape[0], img.shape[1]
#     # 纠正畸变
#     dst1 = cv2.undistort(img, mtx, dist, None, newcameramtx)
#     mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
#     dst2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

#     cv2.imshow('dst1', dst1)
#     cv2.imshow('dst2', dst2)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()
