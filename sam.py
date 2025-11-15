import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import time
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import math
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor


class VideoParticleSegmentor:

    def __init__(self, k=0.09638938232931227):

        self.masks = []

        self.original_img = None
        self.color_img = None
        self.mask_img = None
        self.estimate_img = None

        self.k = k
        self.min_area = 5
        self.max_area = 50000

        self.categories = [0.15, 0.25, 0.5, 1, 3, 5]
        self.categories_counts = [0.0] * (len(self.categories)+1)
        self.categories_areas = [0.0] * (len(self.categories)+1)

        self.mean_diameters = 0

        self.frame_num = 0
        self.frame_area = 0

        self._load_model()
        print('loaded model')
        np.random.seed(3)

    def _load_model(self):
        # select the device for computation
        device = "cuda"
        sam_checkpoint = 'models/sam_vit_h_4b8939.pth'
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=64,
                pred_iou_thresh=0.90,  # 降低IoU阈值以获得更多分割区域
                stability_score_thresh=0.90,  # 降低稳定性阈值
                crop_n_layers=2,
                crop_n_points_downscale_factor=1,
                # min_mask_region_area=10,
                min_mask_region_area=1,
                # box_nms_thresh=0.5,
                output_mode="binary_mask"
            )

    def segment(self, img):
        # copy the img
        self.original_img = img
        # check the image format
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            print('the img format is not rgb')
            return
        # segment particles
        print('start segment')
        self.masks = self.mask_generator.generate(img)
        print(f'len(masks) = {len(self.masks)}')

        return self.masks

    def show_anns(self, is_save=False, borders=True):
        anns = self.masks
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)

        # 1. 生成全透明的RGBA图像（float类型，0-1）
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], 
                    sorted_anns[0]['segmentation'].shape[1], 4), dtype=np.float32)
        img[:, :, 3] = 0  # alpha通道初始化为0（完全透明）

        for ann in sorted_anns:
            m = ann['segmentation'].astype(bool)
            # 随机颜色RGBA，alpha固定0.1
            color_mask = np.concatenate([np.random.random(3), [0]])
            # 用蒙版填充颜色和透明度
            img[m] = color_mask

            if borders:
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # 平滑轮廓
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]

                # 在RGBA图像上画轮廓，颜色为蓝色半透明 (B, G, R, A)
                # 这里直接用红色通道，alpha为0.4
                for contour in contours:
                    cv2.drawContours(img, [contour], -1, (0, 0, 1, 0.4), thickness=1)

        # 把float32的RGBA(0~1)转成uint8的BGRA(0~255)
        img_uint8 = (img * 255).astype(np.uint8)
        self.color_img = cv2.addWeighted(self.original_img, 0.5, img_uint8[:, :, :3], 0.5, 0)
        if not is_save:
            cv2.imshow('Annotations', self.color_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            pass

    def calculate_rotated_rect_iou(self, rect1, rect2):
        """
            计算两个旋转矩形的IoU
            rect1, rect2: cv2.minAreaRect返回的结果 ((cx, cy), (w, h), angle)
        """
        # 创建两个空白mask
        mask1 = np.zeros((2000, 2000), dtype=np.uint8)  # 根据图像大小调整
        mask2 = np.zeros((2000, 2000), dtype=np.uint8)

        # 获取矩形的四个顶点
        box1 = cv2.boxPoints(rect1)
        box2 = cv2.boxPoints(rect2)
        box1 = box1.astype(int)
        box2 = box2.astype(int)

        # 在mask上填充矩形
        cv2.fillPoly(mask1, [box1], 255)
        cv2.fillPoly(mask2, [box2], 255)

        # 计算交集和并集
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0

        iou = intersection / union
        return iou
    
    def calculate_categories_value(self, diameter, area):
        '''
            TODO: optimitize.
            no specific number
        '''
        diameter /= 1000
        if diameter < self.categories[0]:
            self.categories_counts[0] += 1
            self.categories_areas[0] += area
        elif self.categories[0] <= diameter < self.categories[1]:
            self.categories_counts[1] += 1
            self.categories_areas[1] += area
        elif self.categories[1] <= diameter < self.categories[2]:
            self.categories_counts[2] += 1
            self.categories_areas[2] += area
        elif self.categories[2] <= diameter < self.categories[3]:
            self.categories_counts[3] += 1
            self.categories_areas[3] += area
        elif self.categories[3] <= diameter < self.categories[4]:
            self.categories_counts[4] += 1
            self.categories_areas[4] += area
        elif self.categories[4] <= diameter < self.categories[5]:
            self.categories_counts[5] += 1
            self.categories_areas[5] += area
        elif diameter >= self.categories[5]:
            self.categories_counts[6] += 1
            self.categories_areas[6] += area

    def staistic(self):
        if len(self.masks) == 0:
            return

        filtered_large = 0
        filtered_small = 0
        
        self.mask_img = self.original_img.copy()
        # estimate_img = self.original_img.copy()
        # image = self.original_img.copy()

        self.categories_counts[:] = [0 for _ in self.categories_counts]
        self.categories_areas[:] = [0 for _ in self.categories_areas]

        self.frame_num = 0
        self.mean_diameters = 0

        recorded_rects = []
        iou_threshold = 0.3
        image = self.original_img.copy()

        for mask in self.masks:
            area = mask['area']
            if area < self.min_area:
                filtered_small += 1
                continue
            if area > self.max_area:
                filtered_large += 1
                continue
            
            img = mask['segmentation'].astype(np.uint8)*255
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    filtered_small += 1
                    continue
                if area > self.max_area:
                    filtered_large += 1
                    continue

                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = box.astype(int)
                (center_x, center_y), (width, height), angle = rect
                
                is_overlapping = False
                for recorded_rect in recorded_rects:
                    dis = math.sqrt((recorded_rect[0][0]-center_x)**2+(recorded_rect[0][1]-center_y)**2)
                    if dis > 10:
                        # print('too far')
                        continue
                    
                    recorded_area = recorded_rect[1][0] * recorded_rect[1][1]  # 已记录矩形面积
                    iou = self.calculate_rotated_rect_iou(rect, recorded_rect)
                    if area < recorded_area:  # 确保当前矩形更小
                        area_ratio = area / recorded_area  # 理论IoU值
                        if iou >= area_ratio * 0.9:  # 0.9是容错
                            is_overlapping = True  # 当前矩形被包含，应该跳过
                            # print(f"包含在大矩形里面")
                            break
                    elif iou > iou_threshold:
                        is_overlapping = True
                        # print(f"矩形重合度 {iou:.3f} 太高，跳过")
                        break
                # 如果太近，跳过后续计算

                # for debug
                # if not is_overlapping:
                #     cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
                # else:
                #     cv2.drawContours(image, [box], 0, (0, 0, 255), 1)

                # cv2.imshow('img2', image)
                # cv2.waitKey(0)

                if is_overlapping:
                    # print(f"矩形重合度 {iou:.3f} 太高，跳过")
                    continue
                recorded_rects.append(rect)
                cv2.drawContours(self.mask_img, [box], 0, (0, 0, 255), 2)

                # calculate the particle diameter
                equivalent_diameter_px = min(width, height)
                equivalent_diameter_um = equivalent_diameter_px * self.k * 1000
                # statistic
                self.calculate_categories_value(equivalent_diameter_um, area)
                self.frame_num += 1
                self.frame_area += area
                

        # calculate the mean diameter
        if self.frame_num > 0:
            self.mean_diameters = self.mean_diameters/self.frame_num

        # cv2.imshow('mask img', self.mask_img)
        # cv2.waitKey(0)

        return
    
    def save_result(self, video_name, timestamp, n, frame):
        if len(self.masks) == 0:
            return
        
        result_dir = f'./results/{video_name}/{timestamp}'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # 准备数据
        data = {
            'n': n,
            'frame_num': self.frame_num,
            'frame_mean_diameter': self.mean_diameters,
            'total_area': self.frame_area,
        }

        for i in range(len(self.categories)+1):
            if i == 0:
                data[f'<{self.categories[0]}_area'] = self.categories_areas[i]
            elif i == 1:
                data[f'>{self.categories[0]}_area'] = self.categories_areas[i]
            else:
                data[f'>{self.categories[i-1]}_area'] = self.categories_areas[i]

        for i in range(len(self.categories)+1):
            if i == 0:
                data[f'<{self.categories[0]}_num'] = int(self.categories_counts[i])
            elif i == 1:
                data[f'>{self.categories[0]}_num'] = int(self.categories_counts[i])
            else:
                data[f'>{self.categories[i-1]}_num'] = int(self.categories_counts[i])

        df = pd.DataFrame([data])
        csv_path = f'{result_dir}/results_{timestamp}.csv'
        df.to_csv(
            csv_path,
            mode='a' if os.path.exists(csv_path) else 'w',  # 存在则追加，否则新建
            header=not os.path.exists(csv_path),  # 不存在则写表头
            index=True,
            encoding='utf-8-sig'
        )

        # 保存图像
        cv2.imwrite(f"{result_dir}/{n}_frame.png", frame)
        cv2.imwrite(f"{result_dir}/{n}_original.png", self.original_img)
        cv2.imwrite(f"{result_dir}/{n}_masks.png", self.mask_img)
        # cv2.imwrite(f"{result_dir}/{n}_segmented.png", segmented_image)


if __name__ == '__main__':
    mtx_list = [[1.35570138e+04, 0.00000000e+00, 1.30343234e+03],
             [0.00000000e+00, 1.42002596e+04, 1.55487865e+03],
             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    mtx = np.array(mtx_list)
    dist_list = [[-2.66033871e+00, 3.24535255e+02, 1.12022732e-01,-3.05400650e-02, -1.12367025e+04]]
    dist = np.array(dist_list)

    model = VideoParticleSegmentor()

    img = cv2.imread('images/456.png')
    img_copy = img.copy()
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (img.shape[1], img.shape[0]), 0,
                (img.shape[1], img.shape[0]))
    img = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # enhance the contrast
    img = cv2.convertScaleAbs(img, alpha=8, beta=0)
    # resize 
    img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
    print(time.strftime("%H:%M:%S"))
    start = time.time()
    model.segment(img)
    model.staistic()
    model.save_result('test', time.strftime("%H:%M:%S"), 0, img_copy)
    end = time.time()
    print(time.strftime("%H:%M:%S"))
    delta_time = end-start
    print(f'delta time:{int(delta_time/60)}min {delta_time%60}s')
