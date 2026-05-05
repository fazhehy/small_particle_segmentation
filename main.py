import socket
import subprocess
import sys
import os
import numpy as np
import struct
import time
import psutil
import cv2
import struct

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    print(f"memory usage: {mem_mb:.2f} MB")
    return mem_mb

worker = None
conn = None
server = None
sock_path = './tmp/video_socket'

def start_worker():
    global worker, conn, server

    if os.path.exists(sock_path):
        os.remove(sock_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(1)

    worker = subprocess.Popen(
        [sys.executable, 'worker.py', sock_path],
    )
    conn, _ = server.accept()

def send_frame(conn, frame, frame_id, path_str, timestamp_str):
    height, width, channels = frame.shape
    data = frame.tobytes()
    path_bytes = path_str.encode('utf-8')
    timestamp_bytes = timestamp_str.encode('utf-8')
    path_len = len(path_bytes)
    timestamp_len = len(timestamp_bytes)
    header = struct.pack(
        '7I',
        frame_id,
        height,
        width,
        channels,
        len(data),
        path_len,
        timestamp_len
    )
    conn.sendall(header)
    conn.sendall(path_bytes)
    conn.sendall(timestamp_bytes)
    conn.sendall(data)

def receive_result(conn):
    header = conn.recv(8)
    if len(header) < 8:
        return None, None
    frame_id, result_len = struct.unpack('2I', header)
    result_data = b''
    while len(result_data) < result_len:
        chunk = conn.recv(min(4096, result_len - len(result_data)))
        if not chunk:
            break
        result_data += chunk
    return frame_id, result_data.decode()

def stop_worker():
    conn.close()  
    worker.wait()
    server.close()
    if os.path.exists(sock_path):
        os.remove(sock_path)

def main():


    # timestamp = time.strftime("%m.%d-%H:%M:%S")
    # video_path = './data/3.23-3/Video_20260323150428437.mp4'
    # video_name = os.path.basename(os.path.dirname(video_path))
    # frame = cv2.imread('./images/3_frame.png')

    # print(f'======== start {0} ========')
    # print(time.strftime("%H:%M:%S"))
    # start = time.time()
    # start_worker()
    # send_frame(conn, frame, 0, video_name, timestamp)
    # result = None
    # while(result != 'ok'):
    #     _, result = receive_result(conn)
    # stop_worker()
    # end = time.time()
    # print(time.strftime("%H:%M:%S"))
    # delta_time = end-start
    # print(f'delta time:{int(delta_time/60)}min {delta_time%60}s')

    # print(f'========= end {0} =========\n')

    # return

    timestamp = time.strftime("%m.%d-%H:%M:%S")
    video_path = './data/3.23-3/Video_20260323150428437.mp4'
    video_name = os.path.basename(os.path.dirname(video_path))
    # os.makedirs('./data/'+video_name +'/frames', exist_ok=True)
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    


    id = 0
    n = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("视频读取完成")
            break
        id += 1
        if id % 3 == 0 and id > 3879:
            gray_keyframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            min_val, _, _, _ = cv2.minMaxLoc(gray_keyframe)

            # cv2.imshow('keyframe', gray_keyframe)
            # print(f'Frame ID: {id}, Min Pixel Value: {min_val}')
            # if cv2.waitKey(33) & 0xFF == ord('q'):
            #     break
            # cv2.imwrite(f'./data/{video_name}/frames/{id}.jpg', frame)

            if min_val <= 10:
                n += 1
                print(f'======== start {n} ========')
                # print_memory_usage()
                print(time.strftime("%H:%M:%S"))
                start = time.time()
                start_worker()
                send_frame(conn, frame, id, video_name, timestamp)
                result = None
                while(result != 'ok'):
                    _, result = receive_result(conn)
                stop_worker()
                print(time.strftime("%H:%M:%S"))
                delta_time = end-start
                print(f'delta time:{int(delta_time/60)}min {delta_time%60}s')
                # print_memory_usage()
                print(f'========= end {n} =========\n')

if __name__ == "__main__":
    main()