import socket
import sys
import numpy as np
import struct
import time
import cv2
from sam import VideoParticleSegmentor

sock_path = sys.argv[1]

# 连接到主进程
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect(sock_path)
# print("[Worker] Connected to main process!\n")

def receive_frame(conn):
    """
    接收视频帧和附加信息（路径和时间戳）。
    
    协议格式：
    - 4 bytes: frame_id (uint32)
    - 4 bytes: height (uint32)
    - 4 bytes: width (uint32)
    - 4 bytes: channels (uint32)
    - 4 bytes: data_size (uint32)
    - 4 bytes: path_len (uint32)
    - 4 bytes: timestamp_len (uint32)
    - path_len bytes: path string (utf-8)
    - timestamp_len bytes: timestamp string (utf-8)
    - data_size bytes: frame data
    """
    # 先接收固定长度的7个 uint32 元数据，共 7*4 = 28 字节
    header = b''
    while len(header) < 28:
        chunk = conn.recv(28 - len(header))
        if not chunk:
            return None
        header += chunk

    frame_id, height, width, channels, data_size, path_len, timestamp_len = struct.unpack('7I', header)

    # 接收路径字符串
    path_bytes = b''
    while len(path_bytes) < path_len:
        chunk = conn.recv(path_len - len(path_bytes))
        if not chunk:
            return None
        path_bytes += chunk
    path_str = path_bytes.decode('utf-8')

    # 接收时间戳字符串
    timestamp_bytes = b''
    while len(timestamp_bytes) < timestamp_len:
        chunk = conn.recv(timestamp_len - len(timestamp_bytes))
        if not chunk:
            return None
        timestamp_bytes += chunk
    timestamp_str = timestamp_bytes.decode('utf-8')

    # 接收视频帧数据
    data = b''
    while len(data) < data_size:
        chunk = conn.recv(min(8192, data_size - len(data)))
        if not chunk:
            return None
        data += chunk

    # 将字节流转换为 numpy 数组，并reshape成图片格式
    frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, channels))

    return frame_id, path_str, timestamp_str, frame

def send_result(conn, frame_id, message):
    """发送处理结果"""
    data = message.encode()
    header = struct.pack('2I', frame_id, len(data))
    conn.sendall(header)
    conn.sendall(data)

result = receive_frame(client)
if result is not None:
    frame_id, video_name, timestamp, frame = result

    mtx_list = [[1.05876767e+04, 0.00000000e+00, 6.15380489e+02],
                [0.00000000e+00, 1.05010530e+04, 6.02294987e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    mtx = np.array(mtx_list)

    dist_list = [[3.57574264e+00, -2.32625060e+02, -1.02493108e-02, -8.41709324e-02, 7.95046489e+03]]
    dist = np.array(dist_list)

    img = frame
    img_copy = img.copy()
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (img.shape[1], img.shape[0]), 0, (img.shape[1], img.shape[0]))
    img = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # enhance the contrast
    img = cv2.convertScaleAbs(img, alpha=3, beta=0)

    # 1) 去噪（可选）
    denoise = cv2.medianBlur(img, 3)
    # 2) 黑电平压零：三个通道都很低才认为是背景
    thr = 18  # 可调：10~30
    mask_bg = np.all(denoise < thr, axis=2)   # True 表示接近黑背景
    out = denoise.copy()
    out[mask_bg] = [0, 0, 0]
    img = out
    # 1) 轻微去噪（可选）
    img_blur = cv2.medianBlur(img, 3)
    img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    # 2) 提取灰块（阈值可调）
    # 只要亮度 > thr 就当作“非纯黑”
    thr = 20
    _, mask = cv2.threshold(img_blur, thr, 255, cv2.THRESH_BINARY)
    # 3) 连通域过滤：删除小面积灰块
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    min_area = 80   # 关键参数：越大删得越狠
    for i in range(1, num):  # 0是背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == i] = 255
    # 4) 应用掩膜
    out = np.zeros_like(img)
    out[clean_mask == 255] = img[clean_mask == 255]
    # cv2.imshow('preprocessed', out)
    # cv2.waitKey(0)

    # cv2.imshow('undistorted', img)
    # cv2.waitKey(0)
    img = out
    # resize 
    img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)

    model = VideoParticleSegmentor()

    print(time.strftime("%H:%M:%S"))
    start = time.time()
    model.segment(img)
    model.statistic()
    model.save_result(video_name, timestamp, frame_id, img_copy)
    end = time.time()
    print(time.strftime("%H:%M:%S"))
    
    delta_time = end-start
    print(f'delta time:{int(delta_time/60)}min {delta_time%60}s')

    send_result(client, frame_id, 'ok')

client.close()
# print("[Worker] Disconnected")
