import os
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7,8,9")
import time
import cv2
import torch
import pandas as pd
import numpy as np

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

class SAM:
    
    def __init__(self):

        self.min_area = 5
        self.max_area = 50000
        self.iou_threshold = 0.2

        self.mask_generator = None
        self.load_model()

    def load_model(self):

        device = "cuda:0"
        sam_checkpoint = 'models/sam_vit_h_4b8939.pth'
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=64,
                pred_iou_thresh=0.90,  # 降低IoU阈值以获得更多分割区域
                stability_score_thresh=0.90,  # 降低稳定性阈值
                crop_n_layers=1,
                crop_n_points_downscale_factor=1,
                min_mask_region_area=10,
                # min_mask_region_area=1,
                # box_nms_thresh=0.5,
                output_mode="binary_mask"
            )

    def segment(self, image):
        self.original_image = image
        # check the image format
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            print('the image format is not rgb')
            return
        print('Segmenting the image...')
        self.masks = self.mask_generator.generate(image)
        print(f"Generated {len(self.masks)} masks.")
        return self.masks

    def calculate_iou(self, rect1, rect2):

        mask1 = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        mask2 = np.zeros(self.original_image.shape[:2], dtype=np.uint8)

        box1 = cv2.boxPoints(rect1).astype(int)
        box2 = cv2.boxPoints(rect2).astype(int)

        cv2.fillPoly(mask1, [box1], 255)
        cv2.fillPoly(mask2, [box2], 255)

        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        iou = intersection / union
        return iou

    def statistic(self):

        too_small_masks = 0
        too_large_masks = 0

        self.filtered_masks = []
        self.valid_rect = []
        self.valid_rect_data = []

        self.mask_image = self.original_image.copy()

        if len(self.masks) == 0:
            print('No masks generated. Please run segment() first.')
            return None

        for mask in self.masks:
            area = mask['area']
            if area < self.min_area:
                too_small_masks += 1
                continue
            if area > self.max_area:
                too_large_masks += 1
                continue
            
            self.filtered_masks.append(mask)
            image = mask['segmentation'].astype('uint8') * 255
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                contour_area = cv2.contourArea(contour)
                if contour_area < self.min_area:
                    too_small_masks += 1
                    continue
                if contour_area > self.max_area:
                    too_large_masks += 1
                    continue

                rect = cv2.minAreaRect(contour)
                # box = cv2.boxPoints(rect).astype(int)
                (center_x, center_y), (width, height), angle = rect
                
                is_overlapping = False
                for valid_rect in self.valid_rect:
                    distance = np.sqrt((center_x - valid_rect[0][0]) ** 2 + (center_y - valid_rect[0][1]) ** 2)
                    if distance > 100:  
                        continue
                    
                    valid_rect_area = valid_rect[1][0] * valid_rect[1][1]
                    rect_area = width * height
                    iou = self.calculate_iou(rect, valid_rect)
                    if rect_area < valid_rect_area:
                        idea_iou = rect_area / valid_rect_area
                        if iou >= idea_iou*0.9:  
                            is_overlapping = True
                            break
                    elif iou >= self.iou_threshold:  
                        is_overlapping = True
                        break
                
                if is_overlapping:
                    continue

                self.valid_rect.append(rect)
                cv2.drawContours(self.mask_image, [contour], -1, (0, 255, 0), 2)
                self.valid_rect_data.append((area, rect))

        self.get_anns_image(self.filtered_masks)
    
    def get_anns_image(self, masks):
        
        if len(masks) == 0:
            print('No masks generated. Please run segment() first.')
            return None

        anns = masks
        sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], 
                    sorted_anns[0]['segmentation'].shape[1], 4), dtype=np.float32)
        img[:, :, 3] = 0

        for ann in sorted_anns:
            m = ann['segmentation'].astype(bool)

            color_mask = np.concatenate([np.random.random(3), [0]])
            img[m] = color_mask

            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]

            for contour in contours:
                cv2.drawContours(img, [contour], -1, (0, 0, 1, 0.4), thickness=1)

        img_uint8 = (img * 255).astype(np.uint8)
        self.color_img = cv2.addWeighted(self.original_image, 0.5, img_uint8[:, :, :3], 0.5, 0)

    def save_result(self, video_name, timestamp, frame_id, frame):
        result_dir = f'./results/{video_name}/{timestamp}'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        rows = []
        masks_count = len(self.valid_rect_data)
        for i, (area, rect) in enumerate(self.valid_rect_data):
            (center_x, center_y), (width, height), angle = rect
            rows.append({
                "frame_id": frame_id,
                'masks_count': masks_count,
                "mask_id": i,
                "area": area,
                "center_x": center_x,
                "center_y": center_y,
                "width": width,
                "height": height,
                "angle": angle
            })

        df = pd.DataFrame(rows)
        csv_path = f'{result_dir}/{timestamp}.csv'
        file_exists = os.path.exists(csv_path)
        df.to_csv(
            csv_path,
            mode='a' if file_exists else 'w',  # 存在则追加，否则新建
            header=not file_exists,  # 不存在则写表头
            index=False,
            encoding='utf-8-sig'
        )

        cv2.imwrite(f"{result_dir}/{frame_id}_frame.png", frame)
        cv2.imwrite(f"{result_dir}/{frame_id}_original.png", self.original_image)
        cv2.imwrite(f"{result_dir}/{frame_id}_masks.png", self.mask_image)
        cv2.imwrite(f"{result_dir}/{frame_id}_color_masks.png", self.color_img)

if __name__ == "__main__":
    
    calib_file = './calibration/results/mindvision/calibration_result.yml'
    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f'无法打开标定结果文件: {calib_file}')
    
    mtx = fs.getNode("Mat_cam").mat()
    dist = fs.getNode("dist_cam").mat()
    img_size = tuple(fs.getNode("img_size").mat().flatten().astype(int))
    scale = fs.getNode("scale").real()
    proportion = fs.getNode("proportion").real()
    proportion_scaled = fs.getNode("proportion_scaled").real()

    fs.release()

    img = cv2.imread('images/3_frame.png')
    img_copy = img.copy()

    if img_size != img.shape[:2]:
        raise RuntimeError(f'图像尺寸与标定结果不匹配: {img_size} vs {img.shape[:2]}')

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (img.shape[1], img.shape[0]), 0, (img.shape[1], img.shape[0]))
    img = cv2.undistort(img, mtx, dist, None, newcameramtx)

    denoise = cv2.medianBlur(img, 3)
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

    img = out

    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    model = SAM()

    print(time.strftime("%H:%M:%S"))
    start = time.time()
    model.segment(img)
    model.statistic()
    model.save_result('test', time.strftime("%H:%M:%S"), 0, img_copy)
    end = time.time()
    print(time.strftime("%H:%M:%S"))

    delta_time = end-start
    print(f'delta time:{int(delta_time/60)}min {delta_time%60}s')

